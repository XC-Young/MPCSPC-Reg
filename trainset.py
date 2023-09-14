import torch
import numpy as np
import glob, random
import open3d as o3d
from tqdm import tqdm
from utils.utils import *
from utils.r_eval import *
from knn_search import knn_module
from fcgf_model import load_model
from utils.misc import extract_features
import multiprocessing as mp

class generate_trainset:
  def __init__(self):
    self.trainseq = [0,1,2,3,4,5]
    self.valseq = [6,7]
    self.basedir = f'./data/origin_data/kitti_train'
    self.feat_train_dir = f'./data/cache/train'
    make_non_exists_dir(self.feat_train_dir)

    self.load_model()
    self.G = np.load(f'./group_related/Rotation_8.npy')
    self.knn = knn_module.KNN(1)
    self.batchsize = 64

  def loadset(self):
    self.train = {}
    for i in range(8):
      seq = {
        'pc':[],
        'pair':{}
      }
      pair_fns = f'{self.basedir}/{i}/PointCloud/gt.log'
      with open(pair_fns,'r') as f:
        lines = f.readlines()
        pair_num = len(lines)//5
        for k in range(pair_num):
            id0,id1=np.fromstring(lines[k*5],dtype=np.float32,sep=' ')[0:2]
            id0=int(id0)
            id1=int(id1)
            row0=np.fromstring(lines[k*5+1],dtype=np.float32,sep=' ')
            row1=np.fromstring(lines[k*5+2],dtype=np.float32,sep=' ')
            row2=np.fromstring(lines[k*5+3],dtype=np.float32,sep=' ')
            row3=np.fromstring(lines[k*5+4],dtype=np.float32,sep=' ')
            transform=np.stack([row0,row1,row2,row3])
            seq['pair'][f'{id0}-{id1}'] = transform
            if not id0 in seq['pc']:
                seq['pc'].append(id0)
            if not id1 in seq['pc']:
                seq['pc'].append(id1)
      self.train[f'{i}'] = seq

  def gt_match(self):
    #   for seqs in [self.trainseq, self.valseq]:
      for seqs in [self.valseq]:
          for i in seqs:
              seq = self.train[f'{i}']
              savedir = f'{self.feat_train_dir}/{i}/gt_match'
              make_non_exists_dir(savedir)
              for pair,trans in tqdm(seq['pair'].items()):
                  id0,id1=str.split(pair,'-')
                  pc0 = o3d.io.read_point_cloud(f'{self.basedir}/{i}/PointCloud/cloud_bin_{id0}.ply')
                  pc1 = o3d.io.read_point_cloud(f'{self.basedir}/{i}/PointCloud/cloud_bin_{id1}.ply')
                  pc0 = np.array(pc0.points)
                  pc1 = np.array(pc1.points)
                  key0 = np.loadtxt(f'{self.basedir}/{i}/Keypoints/cloud_bin_{id0}Keypoints.txt').astype(np.int)
                  key1 = np.loadtxt(f'{self.basedir}/{i}/Keypoints/cloud_bin_{id1}Keypoints.txt').astype(np.int)
                  key0 = pc0[key0]
                  key1 = pc1[key1]
                  key0 = apply_transform(key0, trans) #align
                  # pair with the filtered keypoints: index in keys
                  dist = np.sum(np.square(key0[:,None,:]-key1[None,:,:]),axis=-1) 
                  # match
                  thres = 0.3*1.5
                  d_min = np.min(dist,axis=1)
                  arg_min = np.argmin(dist,axis=1)
                  m0 = np.arange(d_min.shape[0])[d_min<thres*thres]
                  m1 = arg_min[d_min<thres*thres]
                  pair = np.concatenate([m0[:,None],m1[:,None]],axis=1) #pairnum*2
                  save_fn = f'{savedir}/{id0}_{id1}.npy'
                  np.save(save_fn, pair)

  def load_model(self):
      checkpoint = torch.load('./model/Backbone/best_val_checkpoint.pth')
      config = checkpoint['config']
      Model = load_model(config.model)
      num_feats = 1
      self.model = Model(
          num_feats,
          config.model_n_out,
          bn_momentum=0.05,
          normalize_feature=config.normalize_feature,
          conv1_kernel_size=config.conv1_kernel_size,
          D=3)
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.model.to(self.device)
      self.model.load_state_dict(checkpoint['state_dict'])
      self.model.eval()

  def generate_scan_gfeats(self,pc,key):
      feats = []
      if pc.shape[0]>40000:
          index = np.arange(pc.shape[0])
          np.random.shuffle(index)
          pc = pc[index[0:40000]]
      for gid in range(self.G.shape[0]):
          feats_g = []
          g = self.G[gid]
          #rot the point cloud
          pc_g = pc@g.T
          key_g = key@g.T
          with torch.no_grad():
              pc_g_down, feature_g = extract_features(
                                  self.model,
                                  xyz=pc_g,
                                  voxel_size=0.3,
                                  device=self.device,
                                  skip_check=True)
          feature_g=feature_g.cpu().numpy()
          xyz_down_pcd = o3d.geometry.PointCloud()
          xyz_down_pcd.points = o3d.utility.Vector3dVector(pc_g_down)
          pcd_tree = o3d.geometry.KDTreeFlann(xyz_down_pcd)
          for k in range(key_g.shape[0]):
              [_, idx, _] = pcd_tree.search_knn_vector_3d(key_g[k], 1)
              feats_g.append(feature_g[idx[0]][None,:])
          feats_g=np.concatenate(feats_g,axis=0)#kn*32
          feats.append(feats_g[:,:,None])
      feats = np.concatenate(feats, axis=-1)#kn*32*8
      return feats

  def R2DR_id(self,R):
      min_diff=180
      best_id=0
      for R_id in range(self.G.shape[0]):
          R_diff=compute_R_diff(self.G[R_id],R)
          if R_diff<min_diff:
              min_diff=R_diff
              best_id=R_id
      return best_id

  def DeltaR(self,R,index):
      R_anchor=self.G[index]#3*3
      #R=Rres@Ranc->Rres=R@Ranc.T
      deltaR=R@R_anchor.T
      return quaternion_from_matrix(deltaR)

  def train_feats(self):
    #   batchsavedir = f'{self.feat_train_dir}/train_val_batch/trainset'
    #   make_non_exists_dir(batchsavedir)
      featsavedir = f'{self.feat_train_dir}/train_feats'
      make_non_exists_dir(featsavedir)
      scannum = 0
      all_Rs, all_Rids, all_Rres, all_feats0, all_feats1 = [],[],[],[],[]
      for i in self.trainseq:
          seq = self.train[f'{i}']
          for pair, trans in tqdm(seq['pair'].items()):
              id0,id1=str.split(pair,'-')
              pc0 = np.load(f'{self.basedir}/{i}/PointCloud/cloud_bin_{id0}.npy')
              pc1 = np.load(f'{self.basedir}/{i}/PointCloud/cloud_bin_{id1}.npy')
              key_idx0 = np.loadtxt(f'{self.basedir}/{i}/Keypoints/cloud_bin_{id0}Keypoints.txt').astype(np.int64)
              key_idx1 = np.loadtxt(f'{self.basedir}/{i}/Keypoints/cloud_bin_{id1}Keypoints.txt').astype(np.int64)
              key0 = pc0[key_idx0]
              key1 = pc1[key_idx1]
              R_base = random_rotation_zgroup()
              # gt alignment
              pc0 = apply_transform(pc0, trans) 
              key0 = apply_transform(key0, trans)
              # 1.random z rotation to pc0&pc1 2.group rot to pc1 3.residual rot to pc1
              R_z = random_z_rotation(180)
              R_45 = random_z_rotation(45)
              pc0 = pc0@R_z.T
              pc1 = ((pc1@R_z.T)@R_base.T)@R_45.T
              key0 = key0@R_z.T
              key1 = ((key1@R_z.T)@R_base.T)@R_45.T
              # added rot
              R = R_45@R_base
              R_index = self.R2DR_id(R)
              R_residual = self.DeltaR(R,R_index)
              #gennerate rot feats
              feats0 = self.generate_scan_gfeats(pc0, key0) #5000*32*8
              feats1 = self.generate_scan_gfeats(pc1, key1)

              pt_pair = np.load(f'{self.feat_train_dir}/{i}/gt_match/{id0}_{id1}.npy').astype(np.int32)
              index = np.arange(pt_pair.shape[0])
              np.random.shuffle(index)
              index = index[0:self.batchsize]
              pt_pair = pt_pair[index]
              # paired feats
              feats0 = feats0[pt_pair[:,0],:,:] #64*32*8
              feats1 = feats1[pt_pair[:,1],:,:]
              # save
              all_Rs.append(R[None,:,:])
              all_Rids.append(R_index)
              all_Rres.append(R_residual[None,:])
              all_feats0.append(feats0[None,:,:])
              all_feats1.append(feats1[None,:,:])
              scannum += 1
      np.save(f'{featsavedir}/Rs.npy', np.concatenate(all_Rs,axis=0))
      np.save(f'{featsavedir}/Rids.npy', np.array(all_Rids))
      np.save(f'{featsavedir}/Rres.npy', np.concatenate(all_Rres,axis=0))
      np.save(f'{featsavedir}/feats0.npy', np.concatenate(all_feats0,axis=0))
      np.save(f'{featsavedir}/feats1.npy', np.concatenate(all_feats1,axis=0))
    #   all_Rs = np.concatenate(all_Rs,axis=0)
    #   all_Rids = np.array(all_Rids)
    #   all_Rres = np.concatenate(all_Rres,axis=0)
    #   all_feats0 = np.concatenate(all_feats0,axis=0)
    #   all_feats1 = np.concatenate(all_feats1,axis=0)

  def generate_batches(self, start = 0):
      batchsavedir = f'{self.feat_train_dir}/train_val_batch/trainset'
      make_non_exists_dir(batchsavedir)
      featsavedir = f'{self.feat_train_dir}/train_feats'
      # Generate batch by random combination
      batch_i = start
      all_Rs = np.load(f'{featsavedir}/Rs.npy')
      all_Rids = np.load(f'{featsavedir}/Rids.npy')
      all_Rres = np.load(f'{featsavedir}/Rres.npy')
      all_feats0 = np.load(f'{featsavedir}/feats0.npy')
      all_feats1 = np.load(f'{featsavedir}/feats1.npy')
      scannum = len(all_Rids)
      for i in tqdm(range(scannum)):
          batch_num = random.sample(range(0,scannum),self.batchsize)
          b_feats0s,b_feats1s, b_Rs, b_Rids, b_Rres = [],[],[],[],[]
          for j in range(self.batchsize):
              feats_num = np.random.randint(0,self.batchsize)
              batch_idx = batch_num[j]
              s_R = all_Rs[batch_idx]
              s_Rid = all_Rids[batch_idx]
              s_Rre = all_Rres[batch_idx]
              s_feats0 = all_feats0[batch_idx]
              s_feats0 = s_feats0[feats_num]
              s_feats1 = all_feats1[batch_idx]
              s_feats1 = s_feats1[feats_num]
              b_Rs.append(s_R)
              b_Rids.append(s_Rid)
              b_Rres.append(s_Rre)
              b_feats0s.append(s_feats0)
              b_feats1s.append(s_feats1)
          item = {
            'feats0':torch.from_numpy(np.array(b_feats0s).astype(np.float32)), #before enhanced rot
            'feats1':torch.from_numpy(np.array(b_feats1s).astype(np.float32)), #after enhanced rot
            'R':torch.from_numpy(np.array(b_Rs).astype(np.float32)),
            'true_idx':torch.from_numpy(np.array(b_Rids).astype(np.int)),
            'deltaR':torch.from_numpy(np.array(b_Rres).astype(np.float32))
          }
          torch.save(item,f'{batchsavedir}/{batch_i}.pth',_use_new_zipfile_serialization=False)
          batch_i += 1

  def generate_val_batches(self, vallen = 3000):
      batchsavedir = f'{self.feat_train_dir}/train_val_batch/valset'
      make_non_exists_dir(batchsavedir)
      # generate matches
      matches = []
      for i in self.valseq:
          seq = self.train[f'{i}']
          for pair, trans in tqdm(seq['pair'].items()):
              id0,id1=str.split(pair,'-')
              pair = np.load(f'{self.feat_train_dir}/{i}/gt_match/{id0}_{id1}.npy').astype(np.int32)
              for p in range(pair.shape[0]):
                  matches.append((i,id0,id1,pair[p][0],pair[p][1],trans))
      random.shuffle(matches)
      batch_i=0
      
      for batch_i in tqdm(range(vallen)):
          tup = matches[batch_i]
          scene, id0, id1, pt0, pt1, trans = tup
          pc0 = np.load(f'{self.basedir}/{scene}/PointCloud/cloud_bin_{id0}.npy')
          pc1 = np.load(f'{self.basedir}/{scene}/PointCloud/cloud_bin_{id1}.npy')
          key_idx0 = np.loadtxt(f'{self.basedir}/{scene}/Keypoints/cloud_bin_{id0}Keypoints.txt').astype(np.int64)
          key_idx1 = np.loadtxt(f'{self.basedir}/{scene}/Keypoints/cloud_bin_{id1}Keypoints.txt').astype(np.int64)
          key0 = pc0[key_idx0]
          key1 = pc1[key_idx1]
          R_base = random_rotation_zgroup()
          # gt alignment
          pc0 = apply_transform(pc0, trans) #align
          key0 = apply_transform(key0, trans)
          # 1.random z rotation to pc0&pc1 2.group rot to pc1 3.residual rot to pc1
          R_z = random_z_rotation(180)
          R_45 = random_z_rotation(45)
          pc0 = pc0@R_z.T
          pc1 = ((pc1@R_z.T)@R_base.T)@R_45.T
          key0 = key0@R_z.T
          key1 = ((key1@R_z.T)@R_base.T)@R_45.T
          # added rot
          R = R_45@R_base
          R_index = self.R2DR_id(R)
          R_residual = self.DeltaR(R,R_index)
          #gennerate rot feats
          feats0 = self.generate_scan_gfeats(pc0, key0) #5000*32*8
          feats1 = self.generate_scan_gfeats(pc1, key1)

          feats0 = feats0[int(pt0)] #32*8
          feats1 = feats1[int(pt1)]
          # joint to be a batch
          item={
                  'feats0':torch.from_numpy(feats0.astype(np.float32)), #before enhanced rot
                  'feats1':torch.from_numpy(feats1.astype(np.float32)), #after enhanced rot
                  'R':torch.from_numpy(R.astype(np.float32)),
                  'true_idx':torch.from_numpy(np.array([R_index]).astype(np.int)),
                  'deltaR':torch.from_numpy(R_residual.astype(np.float32))
              }
          # save
          torch.save(item,f'{batchsavedir}/{batch_i}.pth',_use_new_zipfile_serialization=False)
          batch_i += 1
      
              
  def trainval_list(self):
      traindir = f'{self.feat_train_dir}/train_val_batch/trainset'
      valdir = f'{self.feat_train_dir}/train_val_batch/valset'
      trainlist = glob.glob(f'{traindir}/*.pth')
      vallist = glob.glob(f'{valdir}/*.pth')
      save_pickle(range(len(trainlist)), f'{self.feat_train_dir}/train_val_batch/train.pkl')
      save_pickle(range(len(vallist)), f'{self.feat_train_dir}/train_val_batch/val.pkl')      

if __name__=='__main__':
    generator = generate_trainset()
    generator.loadset()
    generator.gt_match()
    generator.train_feats()
    for i in range(3):
        generator.generate_batches(start = len(glob.glob(f'{generator.feat_train_dir}/train_val_batch/trainset/*.pth')))
    generator.generate_val_batches()
    generator.trainval_list()

