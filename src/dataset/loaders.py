"""DataLoader factory."""

from torch.utils.data import DataLoader

from .test_dataset import MultiViewEvalDataset
from .train_dataset import MultiViewXRayDataset


def make_loaders(train_df, val_df, cfg, train_tf, val_tf):
    """Build Phase-2 train and val DataLoaders.

    Training draws a random pair per study; validation is deterministic
    (file order, consecutive windows, single-image studies flagged).
    """
    train_ds = MultiViewXRayDataset(
        train_df, cfg.train_dir, transform=train_tf, num_views=cfg.num_views
    )
    val_ds = MultiViewEvalDataset.from_dataframe(
        val_df, cfg.train_dir, transform=val_tf, num_views=cfg.num_views
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )
    return train_dl, val_dl
