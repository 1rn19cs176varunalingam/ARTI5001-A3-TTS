"""Dataset over the cached features of §1.1.10: (phoneme ids, normalised log-mel)."""
import json, torch
from torch.utils.data import Dataset
from tts.text import PAD_ID


class TTSDataset(Dataset):
    def __init__(self, records_json, mel_dir, mel_stats):
        with open(records_json) as f:
            self.records = json.load(f)
        self.mel_dir, self.mean, self.std = mel_dir, mel_stats["mel_mean"], mel_stats["mel_std"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        mel = torch.load(f"{self.mel_dir}/{r['id']}.pt").float()
        return torch.tensor(r["phonemes"], dtype=torch.long), (mel - self.mean) / self.std


def collate(batch):
    """List of (ids, mel) -> padded ids (B, T_text), true text lengths, padded mels (B, 80, T_mel), true mel lengths."""
    ids_list, mel_list = zip(*batch)
    ids_lens = torch.tensor([len(x) for x in ids_list])
    mel_lens = torch.tensor([m.shape[1] for m in mel_list])
    ids = torch.full((len(batch), int(ids_lens.max())), PAD_ID, dtype=torch.long)
    mels = torch.zeros(len(batch), mel_list[0].shape[0], int(mel_lens.max()))
    for i, (x, m) in enumerate(zip(ids_list, mel_list)):
        ids[i, :len(x)] = x
        mels[i, :, :m.shape[1]] = m
    return ids, ids_lens, mels, mel_lens
