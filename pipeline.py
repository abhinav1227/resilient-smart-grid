import yaml
import pathlib
import torch
import random
import numpy as np
import pandapower.networks as nw
from prefect import task, flow, get_run_logger

# Modular Imports
from data.generate import generate_time_series
from data.preprocess import create_temporal_pyg_data, train_test_split
from models.gcn import DeepPowerSTGAT
from attacks.pgd import pgd_attack
from explanation.integrated_gradients import explain_attack
from detector import calibrate_thresholds, run_ids_audit

def set_seed(seed=77):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# --- TASKS ---

@task(name="1. Extract: AC Physics Simulation", retries=1)
def extract_task(config):
    logger = get_run_logger()
    logger.info(f"Loading {config['data']['grid_case']}...")
    path = "data/storage/raw_grid_data.pt"
    force_extract = config['data'].get('force_extract', False)
    
    # Logic: Skip extraction if data already exists AND we haven't forced a re-extraction
    if pathlib.Path(path).exists() and not force_extract:
        logger.info(f">>> Found existing data at {path}. Skipping extraction task.")
        return path
    
    net = getattr(nw, config['data']['grid_case'])()
    features, targets, ei, ea = generate_time_series(
        net, 
        n_timesteps=config['data']['n_timesteps'],
        seed=config['data']['seed']
    )
    
    torch.save({"x": features, "y": targets, "ei": ei, "ea": ea}, path)
    return path

@task(name="2. Transform: Temporal Graph Construction")
def transform_task(raw_path, config):
    raw = torch.load(raw_path, weights_only=False)
    path = "data/storage/processed_tensors.pt"
    force_transform = config['data'].get('force_transform', False)

    if pathlib.Path(path).exists() and not force_transform:
        logger = get_run_logger()
        logger.info(f">>> Found existing processed data at {path}. Skipping transformation task.")
        return path

    # Skip transformation if processed data already exists AND we haven't forced a re-transformation

    force_transform = config['data'].get('force_transform', False)

    # The Logic: window_size=5 establishes physical inertia
    data_list = create_temporal_pyg_data(
        raw['x'], raw['y'], raw['ei'], raw['ea'], 
        window_size=config['data'].get('window_size', 5)
    )
    
    train, test = train_test_split(data_list, config['data']['train_ratio'])
    torch.save({"train": train, "test": test, "edge_index": raw['ei'], "edge_attr": raw['ea']}, path)
    return path

@task(name="3. Train: Adversarial DeepSTGAT")
def train_task(processed_path, config):
    logger = get_run_logger()
    save_path = pathlib.Path(config['model']['save_path'])
    force_retrain = config['training'].get('force_retrain', False)

    # Skip training if the model exists AND we haven't forced a retrain
    if save_path.exists() and not force_retrain:
        logger.info(f">>> Found existing model at {save_path}. Skipping training task.")
        return str(save_path)

    logger.info(">>> Starting fresh training phase...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load processed data for training
    data = torch.load(processed_path, weights_only=False)
    
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(data['train'], batch_size=config['training']['batch_size'], shuffle=True)
    
    model = DeepPowerSTGAT(
        in_channels=config['model']['in_channels'], 
        hidden_dim=config['model']['hidden_dim'], 
        out_channels=config['model']['out_channels'],
        num_layers=config['model'].get('num_layers', 5)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['lr'])
    criterion = torch.nn.MSELoss()

    # --- Training Loop ---
    for epoch in range(1, config['model']['epochs'] + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Physics-Informed Augmentation (DropEdge)
            mask = torch.rand(batch.edge_index.shape[1], device=device) > 0.05
            batch.edge_index = batch.edge_index[:, mask]
            batch.edge_attr = batch.edge_attr[mask]

            if config['training'].get('adversarial_training'):
                adv_batch = pgd_attack(batch, model, config['attack']['epsilon'], 0.5, 3, criterion)
                out = model(adv_batch)
            else:
                out = model(batch)

            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # Ensure the directory exists before saving
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_path))
    return str(save_path)

@task(name="4. Audit: Security & Explainability", log_prints=True)
def audit_task(model_path, data_path, config):
    logger = get_run_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(data_path, weights_only=False)
    
    # Initialize Model
    model = DeepPowerSTGAT(
        in_channels=config['model']['in_channels'], 
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model'].get('num_layers', 5)
    ).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Cast Tensors for Type-Safety
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long).to(device)
    edge_attr = torch.tensor(data['edge_attr'], dtype=torch.float32).to(device)

    # Verbose Audit
    stats = run_ids_audit(model, data['test'], calibrate_thresholds(data['test'], model, device), device)
    
    # Print the Final Audit Report to terminal
    print("\n" + "="*50)
    print("🛡️ FINAL SECURITY AUDIT REPORT 🛡️")
    print(f"Node Detection: {stats['node_caught']}/{stats['node_total']}")
    print(f"Edge Detection: {stats['edge_caught']}/{stats['edge_total']}")
    print(f"False Alarms:   {stats['false_alarms']}/{stats['clean_total']}")
    print("="*50)

    # XAI
    explain_attack(model, data['test'][0], edge_index, edge_attr, device, save_path=config['explanation']['plot_filename'])
    
    return stats

# --- FLOW ---

@flow(name="End-to-End Resilience Pipeline")
def run_resilience_pipeline(config_path="config.yaml"):
    set_seed(77)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    pathlib.Path("data/storage").mkdir(parents=True, exist_ok=True)
    pathlib.Path("results").mkdir(exist_ok=True)
    
    raw_path = extract_task(config)
    processed_path = transform_task(raw_path, config)
    model_path = train_task(processed_path, config)
    final_report = audit_task(model_path, processed_path, config)
    
    print("Pipeline Success. Audit Report:", final_report)

if __name__ == "__main__":
    run_resilience_pipeline()