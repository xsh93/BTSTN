# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from pypots.data import mcar, masked_fill
from pygrinder import mcar, masked_fill
import torch


class DataReader:
    def __init__(
            self,
            data=None,
            unit=1,
            dropna=False,
            sort=True,
            header=0,
            scaler=None):

        if data is not None:
            data = data if isinstance(data, list) else [data]
            self.dataframe_raw = self.merge_data(data, dropna, sort, header)
            self.dataframe = copy.deepcopy(self.dataframe_raw)
            time_unit = self.dataframe["time"] / unit
            self.dataframe["time"] = time_unit.round(0)
            self.dataframe["time"] = self.dataframe["time"].astype(int)
            self.unit_mx = self.create_unit_mx(self.dataframe)
            self.matrix, self.scaler = self.normalize(self.dataframe, scaler)
        else:
            self.dataframe, self.unit_mx, self.matrix, self.scaler = None, None, None, None

    # -------------------------------------------------------------------------------------
    def __repr__(self):
        obs_num = self.dataframe.shape[0]
        var_num = self.dataframe.shape[1] - 2
        batch_num = len(self.dataframe["batch"].drop_duplicates().values.tolist())
        desc = "[{} observations x {} variates, {} batch in total]".format(obs_num, var_num, batch_num)
        return desc

    # -------------------------------------------------------------------------------------
    @property
    def shape(self):
        _shape = (self.dataframe.shape[0], self.dataframe.shape[1] - 2)
        return _shape

    # -------------------------------------------------------------------------------------
    def merge_data(self, data, dropna=False, sort=True, header=0):
        assert isinstance(data, list), "Wrong data format."
        if isinstance(data[0], pd.DataFrame):
            dataframe = data[0]
            self.format_data(dataframe)
            for dr in data[1:]:
                self.format_data(dr)
                dataframe = pd.concat([dataframe, dr], axis=0)
        else:
            if not os.path.exists(data[0]):
                raise FileExistsError("{} not found!".format(data[0]))
            dataframe = self.read_file(data[0], header=header)
            self.format_data(dataframe)
            for dpath in data[1:]:
                if not os.path.exists(dpath):
                    raise FileExistsError("{} not found!".format(dpath))
                df = self.read_file(dpath, header=header)
                self.format_data(df)
                dataframe = pd.concat([dataframe, df], axis=0)

        if dropna:
            dataframe.dropna(thresh=3, inplace=True)  # deal with missing
        if sort:
            dataframe.sort_values(
                by=[dataframe.columns[0], dataframe.columns[1]],
                ascending=[True, True],
                axis=0,
                inplace=True)  # sort
        dataframe.reset_index(drop=True, inplace=True)
        return dataframe

    # -------------------------------------------------------------------------------------
    @staticmethod
    def read_file(fpath, header=0):
        ext = os.path.splitext(fpath)
        df = None
        if ext[1] == ".csv":
            df = pd.read_csv(fpath, sep=",", header=header)
        elif ext[1] == ".tsv":
            df = pd.read_csv(fpath, sep="\t", header=header)
        elif ext[1] == ".xlsx":
            df = pd.read_excel(fpath)
        return df

    @staticmethod
    def format_data(data):
        assert isinstance(data, pd.DataFrame), "Wrong data format."
        # rename
        data.columns = data.columns.str.replace(data.columns[0], "batch")
        data.columns = data.columns.str.replace(data.columns[1], "time")
        # transform data type
        # data["batch"] = data["batch"].astype(str)
        # data["time"] = data["time"].astype(float)
        # data.loc[:, "batch"] = data["batch"].astype(str)
        # data.loc[:, "time"] = data["time"].astype(float)

    # -------------------------------------------------------------------------------------
    @staticmethod
    def create_unit_mx(dataframe):
        return 1 * np.eye(len(dataframe["time"]), dtype=float)

    # -------------------------------------------------------------------------------------
    @staticmethod
    def normalize(dataframe, method):
        if isinstance(method, str):
            assert method in ("MinMax", "Standard"), "Normalization Method Not Supported!"
            scaler = StandardScaler() if method == "Standard" else MinMaxScaler(feature_range=(0, 1)) if method == "MinMax" else None
            matrix = scaler.fit_transform(dataframe[dataframe.columns[2:]])
        elif method is None:
            scaler = None
            matrix = np.array(dataframe[dataframe.columns[2:]])
        else:
            scaler = method
            matrix = scaler.transform(dataframe[dataframe.columns[2:]])

        return matrix, scaler

    # -------------------------------------------------------------------------------------
    def query(self, batch=None, time=None):
        if batch is not None:
            batch = [batch] if not isinstance(batch, list) else batch
        else:
            batch = self.dataframe["batch"].drop_duplicates().values.tolist()

        if time is not None:
            time = [time] if not isinstance(time, list) else time
        else:
            time = self.dataframe["time"].drop_duplicates().values.tolist()

        dr = DataReader()
        select_index = (self.dataframe["batch"].isin(batch)) & (self.dataframe["time"].isin(time))
        dr.dataframe_raw = self.dataframe_raw[select_index]
        dr.dataframe_raw.reset_index(drop=True, inplace=True)
        dr.dataframe = self.dataframe[select_index]
        dr.dataframe.reset_index(drop=True, inplace=True)
        dr.unit_mx = self.unit_mx[select_index, :]
        dr.unit_mx = self.unit_mx[:, select_index]
        dr.matrix = self.matrix[select_index, :]
        dr.scaler = self.scaler
        return dr

    # -------------------------------------------------------------------------------------
    def batch_pair(self, maxgap=None, batch=None, direction="forward"):
        if batch is not None:
            batch = [batch] if not isinstance(batch, list) else batch
        else:
            batch = self.dataframe["batch"].drop_duplicates().values.tolist()
        assert direction in ["forward", "backward"], ValueError("Incorrect direction.")

        select_batches = self.dataframe["batch"].isin(batch)
        # df = self.dataframe[select_batches]
        unit_mx = self.unit_mx[np.where(select_batches)]
        matrix = self.matrix[np.where(select_batches)]

        pair_df = pd.DataFrame(data=None, columns=["batch", "start", "end", "gap"])
        pair_unit_mx = np.empty(shape=(0, len(self.dataframe["time"])))
        pair_target_mx = np.empty(shape=(0, (len(self.dataframe.columns) - 2)))
        pair_gap_mx = np.empty(shape=(0, 1))

        n, count = 0, 0
        for bt in batch:
            bt_df = self.dataframe[self.dataframe["batch"] == bt]
            bt_df.reset_index(drop=True, inplace=True)

            # time matrix
            bt_time = bt_df["time"].sort_values()
            bt_time_count = len(bt_time)
            bt_time_series = np.array(bt_time)
            current_maxgap = np.max(bt_time_series) - np.min(bt_time_series) if maxgap is None else maxgap

            bt_time_mx1 = dup_cols(np.reshape(bt_time_series, (bt_time_count, 1)), 0, bt_time_count - 1).astype(float)
            bt_time_mx2 = dup_rows(np.reshape(bt_time_series, (1, bt_time_count)), 0, bt_time_count - 1).astype(float)
            subtract_mx = bt_time_mx2 - bt_time_mx1 if direction == "forward" else bt_time_mx1 - bt_time_mx2
            select_index = (subtract_mx >= 0) & (subtract_mx <= current_maxgap)

            # match dataframe
            start_time = bt_time_mx1[select_index]
            end_time = bt_time_mx2[select_index]
            gap = subtract_mx[select_index]
            pair_df_current_batch = pd.concat([pd.Series([bt] * len(gap), dtype=pd.StringDtype()),
                                               pd.Series(start_time, dtype=float),
                                               pd.Series(end_time, dtype=float),
                                               pd.Series(gap, dtype=float)],
                                              axis=1)
            pair_df_current_batch.columns = ["batch", "start", "end", "gap"]
            pair_df = pd.concat([pair_df, pair_df_current_batch], ignore_index=True)

            # match unit_matrix
            unit_index = np.searchsorted(bt_time_series, bt_time_mx1[select_index])
            pair_unit_mx = np.row_stack((pair_unit_mx, unit_mx[count:count + bt_time_count][unit_index]))

            # match target_matrix
            target_index = np.searchsorted(bt_time_series, bt_time_mx2[select_index])
            pair_target_mx = np.row_stack((pair_target_mx, matrix[count:count + bt_time_count][target_index]))
            pair_gap_mx = np.row_stack((pair_gap_mx, gap.reshape(len(gap), 1)))

            count += bt_time_count

        return pair_df, (pair_unit_mx, pair_gap_mx, pair_target_mx)


class SynDataset:
    def __init__(self, data: None):
        self.oridata = None
        self.condata = None
        self.missing_mask = None
        self.indicating_mask = None
        self.features = []

        if data is None:
            pass
        elif isinstance(data, pd.DataFrame):
            assert data.columns[0].lower() == "batch", "Data Error: First column should be <batch>."
            assert data.columns[1].lower() == "time", "Data Error: Second column should be <time>."
            self.oridata = data
            self.oridata.reset_index(drop=True, inplace=True)
            self.features = data.columns[2:]
        else:
            raise ValueError("Incorrect data format, DataFrame is required.")

    # -------------------------------------------------------------------------------------
    def resample(self, noise_std=0, missing_rate=0, missing_vars=None, random_seed=None):
        assert 0 <= noise_std, "Incorrect standard deviation for Gaussian noise."
        assert 0 <= missing_rate < 1, "Incorrect missing rate, should be within 0-1."
        if random_seed is not None:
            np.random.seed(random_seed)

        # data processing (Add Noise & Missing)
        self.condata = copy.deepcopy(self.oridata)
        xdata = self.condata[self.features].to_numpy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        xdata_std = scaler.fit_transform(xdata)
        xdata_std_noise = xdata_std + np.random.randn(*xdata.shape) * noise_std
        xdata = scaler.inverse_transform(xdata_std_noise)

        if missing_vars is not None:
            missing_vars = [missing_vars] if not isinstance(missing_vars, list) else missing_vars
            self.indicating_mask = np.zeros_like(xdata)
            for var in missing_vars:
                features = self.features.tolist()
                if var in features:
                    pos = features.index(var)
                    _, _, _, indicating_mask_col = mcar(xdata[:, pos], missing_rate)
                    self.indicating_mask[:, pos] = indicating_mask_col
                    xdata[indicating_mask_col == 1, pos] = np.nan
            _, _, self.missing_mask, _ = mcar(xdata, 0)
        else:
            xdata_intact, xdata, self.missing_mask, self.indicating_mask = mcar(xdata, missing_rate)
            xdata = masked_fill(xdata, 1 - self.missing_mask, np.nan)
        self.condata[self.features] = xdata

    # -------------------------------------------------------------------------------------
    def save_csv(self, path, file="all"):
        assert file in ("all", "ori", "con"), "File {} not found".format(file)
        save_dir, save_file = os.path.split(path)
        file_name, file_suffix = os.path.splitext(save_file)
        ori_path = os.path.join(save_dir, file_name + "_ori" + file_suffix)
        sam_path = os.path.join(save_dir, file_name + "_con" + file_suffix)

        os.makedirs(save_dir, exist_ok=True)
        if file == "all":
            if self.condata is not None:
                self.oridata.to_csv(ori_path, index=False)
                self.condata.to_csv(sam_path, index=False)
            else:
                self.oridata.to_csv(ori_path, index=False)
        elif file == "ori":
            self.oridata.to_csv(ori_path, index=False)
        else:
            if self.condata is not None:
                self.condata.to_csv(sam_path, index=False)


def dup_rows(mx, idx, dup_num=1):
    return np.insert(mx, [idx + 1] * dup_num, mx[idx], axis=0)


def dup_cols(mx, idx, dup_num=1):
    return np.insert(mx, [idx + 1] * dup_num, mx[:, [idx]], axis=1)


def countdown(count):
    split1 = torch.where(count <= 0, 1, 0)
    split2 = torch.where(count <= 0, 0, 1)
    count -= 1
    return split1, split2, count
