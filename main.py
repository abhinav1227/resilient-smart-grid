import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pandapower.networks as nw
import numpy as np
import logging
import random

# Assuming these are updated modular imports (we will refine these next)
from data.generate import generate_time_series
from data.preprocess import create_temporal_pyg_data, train_test_split
from models.gcn import PowerSTGAT
from attacks.fgsm import constrained_fgsm_attack
from attacks.pgd import pgd_attack
from explanation.integrated_gradients import explain_attack
from utils.graph_utils import get_edge_index

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """
    Locks down all random number generators to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Forces PyTorch to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GridResiliencePipeline:
    def __init__(self, config_path='config.yaml'):
        """
        Initializes the pipeline, loads config, and dynamically sizes the network.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Pipeline initialized. Using device: {self.device}")
        
        # Dynamically load the requested IEEE bus system
        self.grid_case = self.config['data'].get('grid_case', 'case14')
        self.net = getattr(nw, self.grid_case)()  # equialently: nw.case14() for 'case14'
        self.num_buses = len(self.net.bus)
        logger.info(f"Loaded topology for {self.grid_case} with {self.num_buses} buses.")
        
        self.edge_index = get_edge_index(self.net).to(self.device)
        self.model = self._build_model()
        
    def _build_model(self):
        """Constructs the GCN dynamically based on config and grid size."""
        model = PowerSTGAT(
            in_channels=self.config['model']['in_channels'], 
            hidden_dim=self.config['model']['hidden_dim'], 
            out_channels=self.config['model']['out_channels']
        ).to(self.device)
        return model

    def prepare_data(self):
        """Generates or loads data, ensuring it maps correctly to the dynamic bus size."""
        logger.info("Preparing Spatio-Temporal datasets...")
        
        features, targets, edge_index, edge_attr = generate_time_series(
            net=self.net, 
            n_timesteps=self.config['data']['n_timesteps'],
            seed=self.config['data']['seed'],
            save_path=self.config['data'].get('save_path')
        )
        
        # THE UPGRADE: Pass the data through the sliding window function (e.g., 5 timesteps)
        # The Logic: A 5-timestep window is long enough to establish physical inertia 
        # but short enough to keep training computationally light.
        data_list = create_temporal_pyg_data(features, targets, edge_index, edge_attr, window_size=5)
        
        self.train_data, self.test_data = train_test_split(data_list, self.config['data']['train_ratio'])
        
        self.train_loader = DataLoader(self.train_data, batch_size=self.config['training']['batch_size'], shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.config['training']['batch_size'], shuffle=False)
        logger.info(f"Data prepared: {len(self.train_data)} temporal train samples, {len(self.test_data)} temporal test samples.")

    # pgd adversarial training
    def train1(self):
        """Executes the training loop with Randomized PGD Adversarial Training."""
        logger.info("Starting model training...")
        
        adv_training = self.config['training'].get('adversarial_training', False)
        if adv_training:
            logger.info("🛡️ PGD Adversarial Training ENABLED. Building a truly robust model...")
            
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['model']['lr'],
            weight_decay=self.config['model']['weight_decay']
        )
        criterion = nn.MSELoss()
        best_loss = float('inf')
        
        max_eps = self.config['attack']['epsilon']
        # We use a faster PGD configuration for training to save compute time
        train_alpha = self.config['training'].get('pgd_alpha', 0.5) 
        train_iters = self.config['training'].get('pgd_iters', 3)   
        
        for epoch in range(1, self.config['model']['epochs'] + 1):
            self.model.train()
            train_loss = 0.0
            
            for batch in self.train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # 1. Forward pass on clean data
                out_clean = self.model(batch)
                loss_clean = criterion(out_clean, batch.y)
                
                if adv_training:
                    # 2. Generate a Randomized PGD Attack
                    random_eps = np.random.uniform(low=max_eps * 0.1, high=max_eps)
                    
                    for param in self.model.parameters():
                        param.requires_grad = False
                        
                    # We swap FGSM for our new PGD attack generator
                    adv_batch = pgd_attack(
                        data=batch, 
                        model=self.model, 
                        epsilon=random_eps, 
                        alpha=train_alpha,
                        num_iter=train_iters,
                        criterion=criterion,
                        bounds=self.config['attack']['feature_bounds']
                    )
                    
                    for param in self.model.parameters():
                        param.requires_grad = True
                        
                    # 3. Forward pass on the malicious data
                    out_adv = self.model(adv_batch)
                    loss_adv = criterion(out_adv, batch.y)
                    
                    # 4. Combine the losses
                    loss = 0.5 * loss_clean + 0.5 * loss_adv
                else:
                    loss = loss_clean

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(self.train_loader)
            test_loss = self.evaluate(criterion)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
                
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(self.model.state_dict(), self.config['model']['save_path'])
                
        logger.info("Training complete. Best model saved.")

    # fgsm adversarial training
    def train(self):
        """Executes the training loop with Randomized Adversarial Training for robust defense."""
        logger.info("Starting model training...")
        
        adv_training = self.config['training'].get('adversarial_training', False)
        if adv_training:
            logger.info("🛡️ Randomized Adversarial Training ENABLED. Building a robust model...")
            
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['model']['lr'],
            weight_decay=self.config['model']['weight_decay']
        )
        criterion = nn.MSELoss()
        best_loss = float('inf')
        
        # Pull maximum attack strength from config
        max_eps = self.config['attack']['epsilon']
        
        for epoch in range(1, self.config['model']['epochs'] + 1):
            self.model.train()
            train_loss = 0.0
            
            for batch in self.train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Step 1: Forward pass on pristine grid data
                out_clean = self.model(batch)
                loss_clean = criterion(out_clean, batch.y)
                
                if adv_training:
                    # Step 2: Generate a Randomized Attack
                    # We pick a random epsilon between 20% of the max and the absolute max
                    # This ensures the model learns to defend against ALL attack sizes
                    random_eps = np.random.uniform(low=max_eps * 0.2, high=max_eps)
                    
                    for param in self.model.parameters():
                        param.requires_grad = False
                        
                    adv_batch = constrained_fgsm_attack(
                        data=batch, 
                        model=self.model, 
                        epsilon=random_eps, 
                        criterion=criterion,
                        bounds=self.config['attack']['feature_bounds']
                    )
                    
                    for param in self.model.parameters():
                        param.requires_grad = True
                        
                    # Step 3: Forward pass on the malicious data
                    out_adv = self.model(adv_batch)
                    loss_adv = criterion(out_adv, batch.y)
                    
                    # Step 4: Combine the losses
                    loss = 0.5 * loss_clean + 0.5 * loss_adv
                else:
                    loss = loss_clean

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(self.train_loader)
            
            # Step 5: Evaluate strictly on clean data to ensure we haven't ruined normal operations
            test_loss = self.evaluate(criterion)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
                
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(self.model.state_dict(), self.config['model']['save_path'])
                
        logger.info("Training complete. Best model saved.")

    def train1(self):
        """Executes the training loop with proper state management and loss tracking."""
        logger.info("Starting model training...")
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['model']['lr'],
            weight_decay=self.config['model']['weight_decay']
        )
        criterion = nn.MSELoss()
        best_loss = float('inf')
        
        for epoch in range(1, self.config['model']['epochs'] + 1):
            self.model.train()
            train_loss = 0.0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(self.train_loader)
            
            # Simple validation check
            test_loss = self.evaluate(criterion)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
                
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(self.model.state_dict(), self.config['model']['save_path'])
                
        logger.info("Training complete. Best model saved.")

    def evaluate(self, criterion=nn.MSELoss()):
        """Evaluates the model on the test set."""
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = criterion(out, batch.y)
                test_loss += loss.item()
        return test_loss / len(self.test_loader)

    def run_adversarial_evaluation(self):
        """Executes the FGSM attack and calculates dynamically thresholded metrics."""
        logger.info("Initiating Adversarial Evaluation...")
        self.model.load_state_dict(torch.load(self.config['model']['save_path']))
        self.model.eval()
        
        # 1. Establish the Dynamic Baseline (Normal Operations)
        clean_errors = []
        with torch.no_grad():
            for data in self.test_data:
                data = data.to(self.device)
                out = self.model(data)
                error = F.l1_loss(out, data.y).item()
                clean_errors.append(error)
                
        clean_mean = np.mean(clean_errors)
        clean_std = np.std(clean_errors)
        
        # Apply the 3-Sigma Rule to guarantee clean samples fall below the threshold
        dynamic_threshold = clean_mean + (3 * clean_std)
        logger.info(f"Dynamic Threshold: {dynamic_threshold:.4f} (Mean: {clean_mean:.4f} + 3*Std: {clean_std:.4f})")
        
        # 2. Run the Attack Evaluation
        success_count = 0
        n_samples = min(self.config['attack']['num_samples'], len(self.test_data))
        
        def instability_loss(output, target):
             return F.mse_loss(output, target)
        
        for i in range(n_samples):
            sample = self.test_data[i].to(self.device)
            orig_out = self.model(sample)
            orig_error = F.l1_loss(orig_out, sample.y).item()

            adv_sample = pgd_attack(
                data=sample, 
                model=self.model, 
                epsilon=self.config['attack']['epsilon'], 
                alpha=self.config['attack']['alpha'],
                num_iter=self.config['attack']['num_iter'],
                criterion=instability_loss,
                bounds=self.config['attack']['feature_bounds']
            )
            
            # adv_sample = constrained_fgsm_attack(
            #     data=sample, 
            #     model=self.model, 
            #     epsilon=self.config['attack']['epsilon'], 
            #     criterion=instability_loss,
            #     bounds=self.config['attack']['feature_bounds']
            # )
            
            adv_out = self.model(adv_sample)
            adv_error = F.l1_loss(adv_out, sample.y).item()
            
            # The attack is evaluated against the model's specific baseline for this run
            if adv_error > dynamic_threshold and orig_error <= dynamic_threshold:
                success_count += 1
                
        success_rate = (success_count / n_samples) * 100
        logger.info(f"Adversarial Success Rate (ASR): {success_rate:.2f}% across {n_samples} samples.")
        return adv_sample
    
    def generate_explanations(self, adv_sample):
        """Runs Integrated Gradients on the adversarial sample."""
        logger.info("Generating Explainability Heatmap...")
        try:
            # THE LOGIC: The adv_sample is a PyG Data object that already 
            # contains the topological and physical arrays. We extract 
            # edge_index and edge_attr directly from it to prevent missing-variable crashes.
            explain_attack(
                model=self.model, 
                data=adv_sample, 
                edge_index=adv_sample.edge_index, 
                edge_attr=adv_sample.edge_attr,
                device=self.device,
                num_buses=self.num_buses,
                save_path=self.config['explanation']['plot_filename']
            )
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")

if __name__ == "__main__":
    # Orchestration
    set_seed(77)  # Lock the environment first!

    pipeline = GridResiliencePipeline('config.yaml')
    pipeline.prepare_data()
    pipeline.train()
    last_adv_sample = pipeline.run_adversarial_evaluation()
    pipeline.generate_explanations(last_adv_sample)