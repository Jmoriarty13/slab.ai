import torch
from torch.utils.data import DataLoader
from models.global_vit import GlobalViT
from models.local_vit import LocalViT
from models.fusion_head import FusionHead
from dataset.card_dataset import DualScaleCardDataset
from tqdm import tqdm

def train():
    # Load config (you can use YAML loader here)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DualScaleCardDataset(
        df_path="data/labels.csv",
        image_dir="data/images",
        crops_dir="data/patches"
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    global_model = GlobalViT().to(device)
    local_model = LocalViT().to(device)
    fusion_head = FusionHead().to(device)

    optimizer = torch.optim.AdamW(
        list(global_model.parameters()) + 
        list(local_model.parameters()) + 
        list(fusion_head.parameters()), 
        lr=2e-5
    )

    loss_fn = torch.nn.MSELoss()

    for epoch in range(20):
        global_model.train()
        local_model.train()
        fusion_head.train()

        epoch_loss = 0.0
        for batch in tqdm(loader):
            global_img = batch["global_img"].to(device)
            local_patches = batch["local_patches"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            global_feats = global_model(global_img)
            local_feats = local_model(local_patches)
            predictions = fusion_head(global_feats, local_feats)

            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {epoch_loss / len(loader)}")

if __name__ == "__main__":
    train()
