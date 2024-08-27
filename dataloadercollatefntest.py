from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
  def __getitem__(self, i):
    return "A", (1,2,3,4)
  def __len__(self):
    return 10

dataloader = DataLoader(CustomDataset(), batch_size=2)

print("Batches using default collate_fn")
for batch in dataloader:
  print(batch)

def custom_collate_fn(data):
  aList = [d[0] for d in data]
  cList = [d[1] for d in data]
  return aList, cList

dataloader = DataLoader(CustomDataset(), batch_size=2, collate_fn=custom_collate_fn)

print("Batches using custom collate_fn")
for batch in dataloader:
  print(batch)
