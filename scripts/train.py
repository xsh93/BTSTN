# -*- coding: utf-8 -*-
import os
from datetime import datetime
import copy
from tqdm import trange
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from .model import ModuleUnit
from .utils import DataReader, countdown


class BTSTN:
    def __init__(
            self,
            n_features: int,
            n_dims: int,
            g_inner: int,
            g_layers: int,
            d_inner: int = None,
            d_layers: int = None,
            dropout: float = 0,
            d_dropout: float = 0,
            activation: dict = None,
            max_gap: int = None,
            batch_size: int = None,
            epoch: int = None,
            patience: int = None,
            learning_rate: float = None,
            # batch_size_e: int = None,
            # epoch_e: int = None,
            # patience_e: int = None,
            # learning_rate_e: float = None,
            threshold: float = 0,
            gpu_id: int = -1,
            num_workers: int = 0,
            pin_memory: bool = False,
            saving_path: str = None,
            saving_prefix: str = None,
    ):

        self.n_features = n_features
        self.n_dims = n_dims
        self.g_inner = g_inner
        self.g_layers = g_layers
        self.d_inner = d_inner
        self.d_layers = d_layers
        self.dropout = dropout
        self.d_dropout = d_dropout
        self.activation = activation
        self.max_gap = max_gap
        self.batch_size = batch_size
        self.epoch = epoch
        self.patience = patience
        self.learning_rate = learning_rate
        # self.batch_size_e = batch_size_e
        # self.epoch_e = epoch_e
        # self.patience_e = patience_e
        # self.learning_rate_e = learning_rate_e
        self.threshold = threshold
        self.gpu_id = gpu_id
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.saving_path = saving_path
        self.saving_prefix = saving_prefix

        # initialize
        self.criterion = torch.nn.MSELoss()
        self.optimizer = None
        self.model = None
        self.model_monitor = None

        # note: parallely training currently not supported
        self.device = torch.device("cpu") if self.gpu_id == -1 \
            else torch.device("cuda:{}".format(self.gpu_id)) if torch.cuda.is_available() \
            else torch.device("mps:{}".format(self.gpu_id))

        # checking parameters
        assert self.epoch is not None, "[epoch] should not be null."
        self.patience = self.epoch if self.patience is None else self.patience
        assert self.patience <= self.epoch, "[patience] <= [epoch]"
        self.learning_rate = 0.001 if self.learning_rate is None else self.learning_rate
        assert 0 < self.learning_rate <= 1, "0 < [learning_rate] <= 1"
        # self.batch_size_e = self.batch_size if self.batch_size_e is None else self.batch_size_e
        # self.epoch_e = self.epoch if self.epoch_e is None else self.epoch_e
        # self.patience_e = self.patience if self.patience_e is None else self.patience_e
        # self.learning_rate_e = self.learning_rate if self.learning_rate_e is None else self.learning_rate_e

    # -------------------------------------------------------------------------------------
    # Methods:
    # 1. fit()
    # 2. impute()
    # 3. forecast()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    def fit(self, data: DataReader):
        # Step 1: Set modules
        fnum = data.shape[0]
        f_module = ModuleUnit(input_size=fnum, output_size=self.n_dims, bias=False)
        d_module = ModuleUnit(input_size=self.n_dims,
                              output_size=self.n_features,
                              bias=True,
                              hidden_size=self.d_inner,
                              hidden_layers=self.d_layers,
                              activation=None,
                              dropout=self.d_dropout)
        # e_module = ModuleUnit(input_size=self.n_features,
        #                       output_size=self.n_dims,
        #                       bias=True,
        #                       hidden_size=self.d_inner,
        #                       hidden_layers=self.d_layers,
        #                       activation=None,
        #                       dropout=self.d_dropout)
        gf_module = ModuleUnit(input_size=self.n_dims,
                               output_size=self.n_dims,
                               bias=True,
                               hidden_size=self.g_inner,
                               hidden_layers=self.g_layers,
                               activation=self.activation,
                               dropout=self.dropout)
        gb_module = ModuleUnit(input_size=self.n_dims,
                               output_size=self.n_dims,
                               bias=True,
                               hidden_size=self.g_inner,
                               hidden_layers=self.g_layers,
                               activation=self.activation,
                               dropout=self.dropout)

        model = {"F": f_module, "D": d_module, "Gf": gf_module, "Gb": gb_module}
        self._model_to_device(model)
        optimizer = {}
        for k, v in model.items():
            optimizer["{}".format(k)] = torch.optim.Adam(v.parameters(), lr=self.learning_rate)

        # Step 2: Training basic modules
        train_loss, model = self._train(data,
                                        self.max_gap,
                                        self.batch_size,
                                        self.epoch,
                                        self.patience,
                                        model,
                                        optimizer,
                                        "fitting")

        # Step 3: Training encoder
        # optimizer["E"] = torch.optim.Adam(e_module.parameters(), lr=self.learning_rate_e)
        # model["E"] = e_module
        # train_loss_e, model = self._train_e(data, self.batch_size_e, self.epoch_e, self.patience_e, model, optimizer)
        self.model = model

        # Saving models
        if self.saving_path is not None:
            self.save_model(self.saving_path, self.saving_prefix)

        # Setting grad to False
        for k, v in self.model.items():
            for param in v.parameters():
                param.requires_grad = False

        # return train_loss, train_loss_e
        return train_loss

    # -------------------------------------------------------------------------------------
    def impute(
            self,
            data: DataReader,
            max_gap_f: int = None,
            batch_size_f: int = None,
            epoch_f: int = None,
            patience_f: int = None,
            learning_rate_f: float = None,
            inverse: bool = True
    ):
        max_gap_f = self.max_gap if max_gap_f is None else max_gap_f
        batch_size_f = self.batch_size if batch_size_f is None else batch_size_f
        epoch_f = self.epoch if epoch_f is None else epoch_f
        patience_f = self.patience if patience_f is None else patience_f
        learning_rate_f = self.learning_rate if learning_rate_f is None else learning_rate_f
        assert patience_f <= epoch_f, "[patience_e] <= [epoch_e]"
        assert 0 < learning_rate_f <= 1, "0 < [learning_rate_e] <= 1"

        # Step 1: Set F module
        model = copy.deepcopy(self.model)
        fnum = data.shape[0]
        f_module = ModuleUnit(input_size=fnum, output_size=self.n_dims, bias=False)
        model["F"] = f_module
        self._model_to_device(model)
        optimizer = {"F": torch.optim.Adam(model["F"].parameters(), lr=learning_rate_f)}  # Optimizer

        # Step 2: Training basic modules
        train_loss, model = self._train(data,
                                        max_gap_f,
                                        batch_size_f,
                                        epoch_f,
                                        patience_f,
                                        model,
                                        optimizer,
                                        "imputing")

        # Step 3: F-D calculation
        f_module = model["F"].to(self.device)
        d_module = model["D"].to(self.device)

        unit = torch.FloatTensor(data.unit_mx)
        unit = unit.to(self.device)
        fdata = f_module(unit)
        output = d_module(fdata)
        output = output.detach().to("cpu").numpy()
        if inverse and data.scaler is not None:
            output = data.scaler.inverse_transform(output)

        # Step 4: imputation
        impute_data = pd.DataFrame(output, columns=list(data.dataframe.columns[2:]))
        origin_data = data.dataframe[data.dataframe.columns[2:]]
        na_pos = pd.isnull(origin_data)
        impute_data = impute_data[na_pos]

        origin_data = origin_data.fillna(0)
        impute_data = impute_data.fillna(0)
        integrate_data = origin_data + impute_data

        batch = data.dataframe["batch"]
        time = data.dataframe["time"]
        integrate_df = pd.concat([batch, time, integrate_data], axis=1, join="outer")

        return integrate_df

    # -------------------------------------------------------------------------------------
    def monitor(
            self,
            data: DataReader,
            max_gap_f: int = None,
            batch_size_f: int = None,
            epoch_f: int = None,
            patience_f: int = None,
            learning_rate_f: float = None,
    ):
        max_gap_f = self.max_gap if max_gap_f is None else max_gap_f
        batch_size_f = self.batch_size if batch_size_f is None else batch_size_f
        epoch_f = self.epoch if epoch_f is None else epoch_f
        patience_f = self.patience if patience_f is None else patience_f
        learning_rate_f = self.learning_rate if learning_rate_f is None else learning_rate_f
        assert patience_f <= epoch_f, "[patience_e] <= [epoch_e]"
        assert 0 < learning_rate_f <= 1, "0 < [learning_rate_e] <= 1"

        # Step 1: Set F module
        model = copy.deepcopy(self.model)
        fnum = data.shape[0]
        f_module = ModuleUnit(input_size=fnum, output_size=self.n_dims, bias=False)
        model["F"] = f_module
        self._model_to_device(model)
        optimizer = {"F": torch.optim.Adam(model["F"].parameters(), lr=learning_rate_f)}  # Optimizer

        # Step 2: Training basic modules
        train_loss, self.model_monitor = self._train(data,
                                                     max_gap_f,
                                                     batch_size_f,
                                                     epoch_f,
                                                     patience_f,
                                                     model,
                                                     optimizer,
                                                     "monitoring")

    def forecast(self, data: DataReader, stime: list = None, pred_step: int = 1, multi_step: int = 1, inverse: bool = True):
        # max_gap_f = self.max_gap if max_gap_f is None else max_gap_f
        # batch_size_f = self.batch_size if batch_size_f is None else batch_size_f
        # epoch_f = self.epoch if epoch_f is None else epoch_f
        # patience_f = self.patience if patience_f is None else patience_f
        # learning_rate_f = self.learning_rate if learning_rate_f is None else learning_rate_f
        # assert patience_f <= epoch_f, "[patience_e] <= [epoch_e]"
        # assert 0 < learning_rate_f <= 1, "0 < [learning_rate_e] <= 1"

        select_index = []
        if stime is not None:
            for batch, df in data.dataframe.groupby("batch"):
                index = [False] * len(df["time"])
                for start in stime:
                    if start >= 0:
                        if len(df["time"]) < start + 1:
                            raise ValueError("The start time {} exceeds the number of time in {}.".format(start, batch))
                    else:
                        if len(df["time"]) < np.abs(start):
                            raise ValueError("The start time {} exceeds the number of time in {}.".format(start, batch))
                    index[start] = True
                select_index.extend(index)

        # # Step 1: Set F module
        # model = copy.deepcopy(self.model)
        # fnum = data.shape[0]
        # f_module = ModuleUnit(input_size=fnum, output_size=self.n_dims, bias=False)
        # model["F"] = f_module
        # self._model_to_device(model)
        # optimizer = {"F": torch.optim.Adam(model["F"].parameters(), lr=learning_rate_f)}  # Optimizer
        #
        # # Step 2: Training basic modules
        # train_loss, model = self._train(data,
        #                                 max_gap_f,
        #                                 batch_size_f,
        #                                 epoch_f,
        #                                 patience_f,
        #                                 model,
        #                                 optimizer,
        #                                 "forecasting")

        # F-D calculation
        f_module = self.model_monitor["F"].to(self.device)
        d_module = self.model_monitor["D"].to(self.device)
        gf_module = self.model_monitor["Gf"].to(self.device)

        unit = torch.FloatTensor(data.unit_mx[select_index])
        unit = unit.to(self.device)
        fdata = f_module(unit)

        # forecast
        base_info = data.dataframe[["batch", "time"]][select_index]
        base_info.reset_index(drop=True, inplace=True)
        col_names = list(data.dataframe.columns)
        col_names.insert(2, "pred_time")
        forecast_df = pd.DataFrame(data=None, columns=col_names)

        gout = fdata
        for p in range(pred_step):
            for m in range(multi_step):
                identity = gout
                tmp_gout = gf_module(gout)
                gout = tmp_gout + identity

            output = d_module(gout)
            output = output.detach().to("cpu").numpy()
            if inverse and data.scaler is not None:
                output = data.scaler.inverse_transform(output)

            forecast_data_current = pd.DataFrame(output, columns=list(data.dataframe.columns[2:]))
            current_info = copy.deepcopy(base_info)
            current_info.insert(1, "pred_time", current_info["time"] + (p + 1) * multi_step)
            forecast_df_current = pd.concat([current_info, forecast_data_current], axis=1, join="outer")
            forecast_df = pd.concat([forecast_df, forecast_df_current], axis=0)

        forecast_df.sort_values(by=["batch", "time", "pred_time"], ascending=[True, True, True], axis=0, inplace=True)
        forecast_df.reset_index(drop=True, inplace=True)

        return forecast_df

    # -------------------------------------------------------------------------------------
    def save_model(self, opath, model_name=None) -> None:
        model_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') if model_name is None else model_name
        os.makedirs(opath, exist_ok=True)
        for k, v in self.model.items():
            torch.save(v, os.path.join(opath, "{}__{}.pth".format(k, model_name)))
        print("Model saved: {}".format(opath))

    # -------------------------------------------------------------------------------------
    def load_model(self, ipath, model_name=None) -> None:
        assert os.path.exists(ipath), f"Model path {ipath} does not exist."
        try:
            for file in os.listdir(ipath):
                if os.path.isfile(file):
                    if file.split("__")[1] == model_name:
                        model_path = os.path.join(ipath, file)
                        self.model["".format(file.split("__")[0])] = torch.load(model_path)
        except Exception as e:
            raise e
        print("Model loaded successfully")

    # -------------------------------------------------------------------------------------
    def _model_to_device(self, model: dict):
        for mod in model.values():
            mod.to(self.device)

    # -------------------------------------------------------------------------------------
    def _generate_dataloader(self, data: DataReader, max_gap: int = None, batch_size: int = None):
        # Forward dataset
        _, pair_mx = data.batch_pair(max_gap)
        unit = torch.FloatTensor(pair_mx[0])
        gap = torch.FloatTensor(pair_mx[1])
        target = torch.FloatTensor(pair_mx[2])

        # Backward dataset
        _, pair_mx_back = data.batch_pair(max_gap, direction="backward")
        unit_back = torch.FloatTensor(pair_mx_back[0])
        gap_back = torch.FloatTensor(pair_mx_back[1])
        target_back = torch.FloatTensor(pair_mx_back[2])

        # Dataloader
        dataset = TensorDataset(unit, gap, target, unit_back, gap_back, target_back)
        batch_size = len(dataset) if batch_size is None else batch_size
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory)

        return dataloader

    # -------------------------------------------------------------------------------------
    def _train(
            self,
            data: DataReader,
            max_gap: int,
            batch_size: int,
            epoch: int,
            patience: int,
            model: dict,
            optimizer: dict,
            process="fitting"):

        patience_total = patience
        train_loader = self._generate_dataloader(data, max_gap, batch_size)
        best_loss = float("inf")
        best_model = None
        loss_epoch = pd.DataFrame(data=None, columns=["Epoch", "Loss"])
        with trange(epoch, ncols=100) as t:
            for epo in t:
                t.set_description("==> {}".format(process))
                running_loss = 0
                for unit, gap, target, unit_back, gap_back, target_back in train_loader:
                    output, ground = self._train_flow(unit, gap, target, model, direction="forward")  # Forward
                    output_back, ground_back = self._train_flow(unit_back, gap_back, target_back, model,
                                                                direction="backward")  # Backward

                    # Update
                    for k, v in optimizer.items():
                        v.zero_grad()
                    output_cat = torch.cat([output, output_back], dim=0)
                    ground_cat = torch.cat([ground, ground_back], dim=0)
                    loss = self.criterion(output_cat, ground_cat)
                    loss.backward()
                    for k, v in optimizer.items():
                        v.step()

                    # Loss
                    running_loss += loss.item()
                t.set_postfix(current_loss="{:.5f}".format(running_loss), minimum_loss="{:.5f}".format(best_loss))
                loss_epoch.loc[epo] = [epo + 1, running_loss]

                # Training stops if the loss of all training modules is less than the threshold
                if running_loss <= self.threshold:
                    break

                # Patience: Early stopping
                if patience == 0:
                    break
                else:
                    if running_loss < best_loss:
                        best_loss = running_loss
                        best_model = copy.deepcopy(model)
                        patience = patience_total - 1
                    else:
                        patience -= 1
                        if patience == 0:
                            break

        return loss_epoch, best_model

    # -------------------------------------------------------------------------------------
    def _train_flow(self, x, gap, y, model, direction="forward"):
        assert direction in ["forward", "backward"], ValueError("Incorrect direction.")
        self._model_to_device(model)
        max_gap = gap[:, 0].max()

        # F-D
        x = x.to(self.device)
        sp1, sp2, next_gap = countdown(gap)
        sp1 = sp1.to(self.device)
        sp2 = sp2.to(self.device)
        fdata = model["F"](x)
        dout = torch.multiply(fdata, sp1)

        identity = fdata
        gout = model["Gf"](fdata) if direction == "forward" else model["Gb"](fdata)
        gout += identity
        gout = torch.multiply(gout, sp2)

        # G-loops
        for _ in np.arange(0, max_gap):
            sp1, sp2, next_gap = countdown(next_gap)
            sp1 = sp1.to(self.device)
            sp2 = sp2.to(self.device)
            dout += torch.multiply(gout, sp1)
            identity = gout
            tmp_gout = model["Gf"](gout) if direction == "forward" else model["Gb"](gout)
            tmp_gout += identity
            gout = torch.multiply(tmp_gout, sp2)

        # Checking steps
        assert torch.equal(gout, torch.zeros_like(gout)), "Wrong steps!"

        # Output
        output = model["D"](dout)

        # Missing values do not back-propagate
        ground = copy.deepcopy(y)
        nan_pos = torch.where(torch.isnan(ground))
        tmpout = output.detach().to("cpu")
        for i in range(len(nan_pos[0])):
            ground[nan_pos[0][i], nan_pos[1][i]] = tmpout[nan_pos[0][i], nan_pos[1][i]]
        ground = ground.to(self.device)
        return output, ground

    # -------------------------------------------------------------------------------------
    # def _train_e(self, data: DataReader, batch_size: int, epoch: int, patience: int, model: dict, optimizer: dict):
    #     patience_total = patience
    #     unit = torch.FloatTensor(data.unit_mx).to(self.device)
    #     features = model["F"](unit)
    #     predict_value = model["D"](features)
    #     predict_value = predict_value.detach().to("cpu").numpy()
    #     features = features.detach()
    #
    #     observe_value = data.matrix
    #     missing = np.isnan(data.matrix)
    #     merged_data = np.multiply(observe_value, np.where(missing, 0, 1)) + np.multiply(predict_value,
    #                                                                                     np.where(missing, 1, 0))
    #     merged_data = torch.FloatTensor(merged_data).to(self.device)
    #
    #     # Dataset
    #     dataset = TensorDataset(merged_data, features)
    #     batch_size = len(dataset) if batch_size is None else batch_size
    #     dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
    #                             num_workers=self.num_workers, pin_memory=self.pin_memory)
    #
    #     best_loss = float("inf")
    #     best_model = None
    #     loss_epoch = pd.DataFrame(data=None, columns=["Epoch", "Loss"])
    #     with trange(epoch) as t:
    #         for epo in t:
    #             t.set_description("fitting E")
    #             running_loss = 0
    #             for x, y in dataloader:
    #                 output = model["E"](x)
    #                 # Update
    #                 optimizer["E"].zero_grad()
    #                 loss = self.criterion(output, y)
    #                 loss.backward()
    #                 optimizer["E"].step()
    #
    #                 # Loss
    #                 running_loss += loss.item()
    #             t.set_postfix(loss="{:.5f}".format(running_loss))
    #             loss_epoch.loc[epo] = [epo + 1, running_loss]
    #
    #             # Training stops if the loss of all training modules is less than the threshold
    #             if running_loss <= self.threshold:
    #                 break
    #
    #             # Patience: Early stopping
    #             if patience == 0:
    #                 break
    #             else:
    #                 if running_loss < best_loss:
    #                     best_loss = running_loss
    #                     best_model = copy.deepcopy(model)
    #                     patience = patience_total - 1
    #                 else:
    #                     patience -= 1
    #                     if patience == 0:
    #                         break
    #
    #     return loss_epoch, best_model
