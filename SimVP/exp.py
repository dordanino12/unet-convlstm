import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
from model import SimVP
from tqdm import tqdm
from API import *
from utils import *
import math
import torch.nn.functional as F


class SimVPBinLoss(torch.nn.Module):
    """
    Custom Bin Loss for SimVP.
    Calculates weights based on the physical velocity distribution (m/s).
    """

    def __init__(self, dataset_obj, unmasked_weight_factor=0.1):
        super(SimVPBinLoss, self).__init__()
        self.dataset = dataset_obj
        self.unmasked_weight_factor = unmasked_weight_factor

        # Dynamically extract bin boundaries from SimVP dataset limits
        self.bin_min = -float(dataset_obj.max_neg_val)
        self.bin_max = float(dataset_obj.max_pos_val)
        self.bin_width = 0.1
        self.num_bins = int(math.ceil((self.bin_max - self.bin_min) / self.bin_width))

    def forward(self, y_pred, y, mask=None, debug_bins=False):
        abs_diff = (y_pred - y).abs()

        # Denormalize Y to physical velocity (m/s) using SimVP's scale factor
        y_denorm = y * self.dataset.scale
        y_denorm = y_denorm.clamp(self.bin_min, self.bin_max - 1e-6)

        # Handle mask broadcasting (SimVP has 2 channels for y, 1 channel for mask)
        if mask is not None:
            mask_expanded = mask.expand_as(y_denorm)
            mask_for_bins = mask_expanded > 0.5
            if mask_for_bins.sum() == 0:
                mask_for_bins = None
        else:
            mask_for_bins = None
            mask_expanded = None

        if mask_for_bins is not None:
            y_flat = y_denorm[mask_for_bins]
        else:
            y_flat = y_denorm.flatten()

        # Count pixels in each bin
        bin_counts = torch.zeros(self.num_bins, device=y.device)
        for i in range(self.num_bins):
            bin_start = self.bin_min + i * self.bin_width
            bin_end = bin_start + self.bin_width
            bin_mask = (y_flat >= bin_start) & (y_flat < bin_end)
            bin_counts[i] = bin_mask.sum().float()

        # Calculate inverse frequency weights
        total_pixels = y_flat.numel()
        bin_weights = torch.zeros(self.num_bins, device=y.device)
        non_empty = bin_counts > 0
        bin_weights[non_empty] = total_pixels / (bin_counts[non_empty] + 1e-8)

        # Normalize weights
        if non_empty.any():
            bin_weights[non_empty] = bin_weights[non_empty] / (bin_weights[non_empty].mean() + 1e-8)

        bin_weights = torch.clamp(bin_weights, max=100.0)

        # Assign weights to each pixel
        pixel_bin_weights = torch.zeros_like(y_denorm)
        for i in range(self.num_bins):
            bin_start = self.bin_min + i * self.bin_width
            bin_end = bin_start + self.bin_width
            bin_mask_pixel = (y_denorm >= bin_start) & (y_denorm < bin_end)
            pixel_bin_weights[bin_mask_pixel] = bin_weights[i]

        # Calculate final weighted L1 loss
        if mask is not None:
            combined_weight = pixel_bin_weights * mask_expanded.float()
            denom = combined_weight.sum()
            if denom < 1e-8:
                # Requires grad to prevent graph breaking if empty
                weighted_l1 = torch.zeros((), device=y.device, requires_grad=True)
            else:
                numerator = (abs_diff * combined_weight).sum()
                weighted_l1 = numerator / (denom + 1e-8)
        else:
            weighted_l1 = (abs_diff * pixel_bin_weights).sum() / (pixel_bin_weights.sum() + 1e-8)

        return weighted_l1


class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

    def _build_model(self):
        args = self.args
        self.model = SimVP(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T).to(self.device)

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        # Toggle here: True uses Bin Loss, False uses standard MSE
        self.use_bin_loss = getattr(self.args, 'use_bin_loss', True)

        if self.use_bin_loss:
            print_log("[INFO] Using Custom Bin Loss")
            # We pass the train_loader dataset so it can read max/min limits
            self.criterion = SimVPBinLoss(dataset_obj=self.train_loader.dataset)
        else:
            print_log("[INFO] Using standard MSE Loss")
            self.criterion = torch.nn.MSELoss(reduction='none')

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)

        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)

            for batch_data in train_pbar:
                self.optimizer.zero_grad()

                # Unpack batch (x, y, mask)
                if len(batch_data) == 3:
                    batch_x, batch_y, batch_mask = batch_data
                    batch_x, batch_y, batch_mask = batch_x.to(self.device), batch_y.to(self.device), batch_mask.to(
                        self.device)

                    # Check if we should ignore the mask based on args
                    if not getattr(self.args, 'use_mask', True):
                        batch_mask = None
                else:
                    batch_x, batch_y = batch_data
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    batch_mask = None

                pred_y = self.model(batch_x)

                # --- Loss Calculation Toggle ---
                if hasattr(self, 'use_bin_loss') and self.use_bin_loss:
                    # Bin Loss internally handles the mask and returns a scalar
                    loss = self.criterion(pred_y, batch_y, mask=batch_mask)
                else:
                    # Standard MSE Loss masking logic
                    loss_raw = self.criterion(pred_y, batch_y)  # [B, T, C, H, W]
                    if batch_mask is not None:
                        mask_expanded = batch_mask.expand_as(loss_raw)
                        masked_loss = loss_raw * (mask_expanded > 0.5).float()
                        loss = masked_loss.sum() / (mask_expanded > 0.5).sum().clamp(min=1e-8)
                    else:
                        loss = loss_raw.mean()
                # -------------------------------

                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            train_loss = np.average(train_loss)

            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader)
                    if epoch % (args.log_step * 100) == 0:
                        self._save(name=str(epoch))
                print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
                    epoch + 1, train_loss, vali_loss))
                recorder(vali_loss, self.model, self.path)

        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader):
        self.model.eval()
        # Added masks_lst to collect masks across batches
        preds_lst, trues_lst, masks_lst, total_loss = [], [], [], []
        vali_pbar = tqdm(vali_loader)

        for i, batch_data in enumerate(vali_pbar):
            if i * batch_data[0].shape[0] > 1000:
                break

            # Unpack batch (x, y, mask)
            if len(batch_data) == 3:
                batch_x, batch_y, batch_mask = batch_data
                batch_x, batch_y, batch_mask = batch_x.to(self.device), batch_y.to(self.device), batch_mask.to(
                    self.device)

                # Check if we should ignore the mask based on args
                if not getattr(self.args, 'use_mask', True):
                    batch_mask = None

                # Store the mask to pass it to the metric function later
                if batch_mask is not None:
                    masks_lst.append(batch_mask.detach().cpu().numpy())
            else:
                batch_x, batch_y = batch_data
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                batch_mask = None

            pred_y = self.model(batch_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                pred_y, batch_y], [preds_lst, trues_lst]))

            # --- Loss Calculation Toggle ---
            if hasattr(self, 'use_bin_loss') and self.use_bin_loss:
                # Bin Loss internally handles the mask and returns a scalar
                loss = self.criterion(pred_y, batch_y, mask=batch_mask)
            else:
                # Standard MSE Loss masking logic
                loss_raw = self.criterion(pred_y, batch_y)  # [B, T, C, H, W]
                if batch_mask is not None:
                    mask_expanded = batch_mask.expand_as(loss_raw)
                    masked_loss = loss_raw * (mask_expanded > 0.5).float()
                    loss = masked_loss.sum() / (mask_expanded > 0.5).sum().clamp(min=1e-8)
                else:
                    loss = loss_raw.mean()
            # -------------------------------

            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.item()))
            total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)

        # Build the full mask tensor from all batches if available
        masks = np.concatenate(masks_lst, axis=0) if len(masks_lst) > 0 else None

        # Rescale values back to original physical velocity range
        scale = vali_loader.dataset.scale
        preds = preds * scale
        trues = trues * scale

        # Pass the mask array to the metric function (set standard param to False if data is already rescaled,
        # but keep return_ssim_psnr True to unpack 4 variables)
        mse, mae, ssim, psnr = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, True, mask=masks)
        print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        self.model.train()
        return total_loss

    def test(self, args):
        self.model.eval()
        # Added masks_lst to collect masks across batches
        inputs_lst, trues_lst, preds_lst, masks_lst = [], [], [], []

        for batch_data in self.test_loader:
            # Unpack batch (x, y, mask) or (x, y)
            if len(batch_data) == 3:
                batch_x, batch_y, batch_mask = batch_data
                batch_x = batch_x.to(self.device)

                # Check if we should ignore the mask based on args
                if not getattr(self.args, 'use_mask', True):
                    batch_mask = None

                # Store the mask to pass it to the metric function later
                if batch_mask is not None:
                    masks_lst.append(batch_mask.detach().cpu().numpy())
            else:
                batch_x, batch_y = batch_data
                batch_x = batch_x.to(self.device)
                batch_mask = None

            pred_y = self.model(batch_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        # Build the full mask tensor from all batches if available
        masks = np.concatenate(masks_lst, axis=0) if len(masks_lst) > 0 else None

        # Rescale values back to original physical velocity range for the test set
        scale = self.test_loader.dataset.scale
        preds = preds * scale
        trues = trues * scale

        folder_path = self.path + '/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Pass the mask array to the metric function
        mse, mae, ssim, psnr = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std, True,
                                      mask=masks)
        print_log('mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))

        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return mse