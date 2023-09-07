from customdataset import CustomDataset_distill
from torch.utils.data import DataLoader
import pickle


with open('../../data/ix_tr_s1.pickle', 'rb') as f:
    ix_tr_s1 = pickle.load(f)
with open('../../data/ix_vl_s1.pickle', 'rb') as f:
    ix_vl_s1 = pickle.load(f)
with open('../../data/ix_ts_s1.pickle', 'rb') as f:
    ix_ts_s1 = pickle.load(f)
with open('../../data/iy_tr_s1.pickle', 'rb') as f:
    iy_tr_s1 = pickle.load(f)
with open('../../data/iy_vl_s1.pickle', 'rb') as f:
    iy_vl_s1 = pickle.load(f)
with open('../../data/iy_ts_s1.pickle', 'rb') as f:
    iy_ts_s1 = pickle.load(f)

    
with open('../../data/vx_tr_s1.pickle', 'rb') as f:
    vx_tr_s1 = pickle.load(f)
with open('../../data/vx_vl_s1.pickle', 'rb') as f:
    vx_vl_s1 = pickle.load(f)
with open('../../data/vx_ts_s1.pickle', 'rb') as f:
    vx_ts_s1 = pickle.load(f)
with open('../../data/vy_tr_s1.pickle', 'rb') as f:
    vy_tr_s1 = pickle.load(f)
with open('../../data/vy_vl_s1.pickle', 'rb') as f:
    vy_vl_s1 = pickle.load(f)
with open('../../data/vy_ts_s1.pickle', 'rb') as f:
    vy_ts_s1 = pickle.load(f)


with open('../../data/ix_tr_s2.pickle', 'rb') as f:
    ix_tr_s2 = pickle.load(f)
with open('../../data/ix_vl_s2.pickle', 'rb') as f:
    ix_vl_s2 = pickle.load(f)
with open('../../data/ix_ts_s2.pickle', 'rb') as f:
    ix_ts_s2 = pickle.load(f)
with open('../../data/iy_tr_s2.pickle', 'rb') as f:
    iy_tr_s2 = pickle.load(f)
with open('../../data/iy_vl_s2.pickle', 'rb') as f:
    iy_vl_s2 = pickle.load(f)
with open('../../data/iy_ts_s2.pickle', 'rb') as f:
    iy_ts_s2 = pickle.load(f)

    
with open('../../data/vx_tr_s2.pickle', 'rb') as f:
    vx_tr_s2 = pickle.load(f)
with open('../../data/vx_vl_s2.pickle', 'rb') as f:
    vx_vl_s2 = pickle.load(f)
with open('../../data/vx_ts_s2.pickle', 'rb') as f:
    vx_ts_s2 = pickle.load(f)
with open('../../data/vy_tr_s2.pickle', 'rb') as f:
    vy_tr_s2 = pickle.load(f)
with open('../../data/vy_vl_s2.pickle', 'rb') as f:
    vy_vl_s2 = pickle.load(f)
with open('../../data/vy_ts_s2.pickle', 'rb') as f:
    vy_ts_s2 = pickle.load(f)


with open('../../data/ix_tr_t.pickle', 'rb') as f:
    ix_tr_t = pickle.load(f)
with open('../../data/ix_vl_t.pickle', 'rb') as f:
    ix_vl_t = pickle.load(f)
with open('../../data/ix_ts_t.pickle', 'rb') as f:
    ix_ts_t = pickle.load(f)
with open('../../data/iy_tr_t.pickle', 'rb') as f:
    iy_tr_t = pickle.load(f)
with open('../../data/iy_vl_t.pickle', 'rb') as f:
    iy_vl_t = pickle.load(f)
with open('../../data/iy_ts_t.pickle', 'rb') as f:
    iy_ts_t = pickle.load(f)

    
with open('../../data/vx_tr_t.pickle', 'rb') as f:
    vx_tr_t = pickle.load(f)
with open('../../data/vx_vl_t.pickle', 'rb') as f:
    vx_vl_t = pickle.load(f)
with open('../../data/vx_ts_t.pickle', 'rb') as f:
    vx_ts_t = pickle.load(f)
with open('../../data/vy_tr_t.pickle', 'rb') as f:
    vy_tr_t = pickle.load(f)
with open('../../data/vy_vl_t.pickle', 'rb') as f:
    vy_vl_t = pickle.load(f)
with open('../../data/vy_ts_t.pickle', 'rb') as f:
    vy_ts_t = pickle.load(f)

def dataloader():
    source1_train = CustomDataset_distill(ix_tr_s1, vx_tr_s1, iy_tr_s1)
    source1_train_dataloader = DataLoader(source1_train, batch_size=32, shuffle=True, drop_last=True )
    source1_valid = CustomDataset_distill(ix_vl_s1, vx_vl_s1, iy_vl_s1)
    source1_valid_dataloader = DataLoader(source1_valid, batch_size=1, shuffle=True, drop_last=False )
    source1_test = CustomDataset_distill(ix_ts_s1, vx_ts_s1, iy_ts_s1)
    source1_test_dataloader = DataLoader(source1_test, batch_size=1, shuffle=False, drop_last=False )

    source2_train = CustomDataset_distill(ix_tr_s2, vx_tr_s2, iy_tr_s2)
    source2_train_dataloader = DataLoader(source2_train, batch_size=32, shuffle=True, drop_last=True )
    source2_valid = CustomDataset_distill(ix_vl_s2, vx_vl_s2, iy_vl_s2)
    source2_valid_dataloader = DataLoader(source2_valid, batch_size=1, shuffle=True, drop_last=False )
    source2_test = CustomDataset_distill(ix_ts_s2, vx_ts_s2, iy_ts_s2)
    source2_test_dataloader = DataLoader(source2_test, batch_size=1, shuffle=False, drop_last=False )

    target_train = CustomDataset_distill(ix_tr_t, vx_tr_t, iy_tr_t)
    target_train_dataloader = DataLoader(target_train, batch_size=32, shuffle=True, drop_last=True )
    target_valid = CustomDataset_distill(ix_vl_t, vx_vl_t, iy_vl_t)
    target_valid_dataloader = DataLoader(target_valid, batch_size=1, shuffle=True, drop_last=False )
    target_test = CustomDataset_distill(ix_ts_t, vx_ts_t, iy_ts_t)
    target_test_dataloader = DataLoader(target_test, batch_size=1, shuffle=False, drop_last=False )


    return source1_train_dataloader,source1_valid_dataloader,source1_test_dataloader,source2_train_dataloader,source2_valid_dataloader,source2_test_dataloader,target_train_dataloader,target_valid_dataloader,target_test_dataloader
