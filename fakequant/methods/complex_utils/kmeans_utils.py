
import torch
DEBUG_TEST=False
class KMeansQuantizerTorch:
    def __init__(self, n_clusters=8, max_iter=10):
        """
        K-Means 聚类量化器（基于 PyTorch 实现）
        :param n_clusters: 聚类的簇数（量化级别）
        :param max_iter: 最大迭代次数
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers = None

    def fit(self, weight):
        """
        对权重进行 K-Means 聚类
        :param weight: 输入权重张量 (torch.Tensor)
        :return: None
        """
        # 将权重展平为一维
        weight_flat = weight.reshape(-1, 1)

        # 初始化聚类中心（随机选择 n_clusters 个点作为初始中心）
        # indices = torch.randperm(weight_flat.size(0))[:self.n_clusters]
        # self.cluster_centers = weight_flat[indices].clone()
        
        # 初始化聚类中心（等间距划分）
        weight_min, weight_max = weight_flat.min(), weight_flat.max()
        self.cluster_centers = torch.linspace(weight_min, weight_max, self.n_clusters, device=weight.device).reshape(-1, 1)

        
        # K-Means 聚类迭代
        for _ in range(self.max_iter):
            # 计算每个点到聚类中心的距离
            distances = torch.cdist(weight_flat, self.cluster_centers)
            # 分配每个点到最近的聚类中心
            labels = torch.argmin(distances, dim=1)

            # 更新聚类中心
            for k in range(self.n_clusters):
                if (labels == k).sum() > 0:  # 如果某个簇有点
                    self.cluster_centers[k] = weight_flat[labels == k].mean()

    def quantize(self, weight):
        """
        对权重进行量化
        :param weight: 输入权重张量 (torch.Tensor)
        :return: 量化后的权重张量
        """
        # 将权重展平为一维
        weight_flat = weight.reshape(-1, 1)

        # 计算每个点到聚类中心的距离
        distances = torch.cdist(weight_flat, self.cluster_centers)
        # 分配每个点到最近的聚类中心
        labels = torch.argmin(distances, dim=1)
        # 根据簇索引替换为对应的聚类中心值
        quantized_weight_flat = self.cluster_centers[labels]

        # 恢复为原始形状
        quantized_weight = quantized_weight_flat.reshape(weight.shape)
        return quantized_weight

    def dequantize(self, quantized_weight):
        """
        对量化后的权重进行反量化
        :param quantized_weight: 量化后的权重张量
        :return: 反量化后的权重张量
        """
        # 直接返回量化权重，因为量化值已经是聚类中心
        return quantized_weight
    
test_quantizer=KMeansQuantizerTorch(n_clusters=8, max_iter=3)

class RowWiseKMeansQuantizerTorch:
    def __init__(self, n_clusters=8, group_size=-1,max_iter=3):
        """
        按行 K-Means 聚类量化器（基于 PyTorch 实现）
        :param n_clusters: 聚类的簇数（量化级别）
        :param group_size: 每个块的大小，默认为 -1 表示不分块
        """
        self.n_clusters = n_clusters
        self.row_cluster_centers = []
        self.group_size = group_size
        self.max_iter = max_iter
        


    def fit(self, weight):
        """
        对权重矩阵按行进行 K-Means 聚类
        :param weight: 输入权重张量 (torch.Tensor)
        :return: None
        """
        w_size = weight.size(-1)

        if self.group_size < 0:
            denominator = abs(self.group_size)
            assert w_size % denominator == 0, f"Weight size {w_size} must be divisible by {denominator}"
            group_size = w_size // denominator
        else:
            group_size = self.group_size
            #assert w_size % group_size == 0, f"Weight size {w_size} must be divisible by group_size {group_size}"

        weight=weight.reshape(-1,group_size)
        self.row_cluster_centers = []
        for row in weight:
            # 初始化聚类中心
            row_min, row_max = row.min(), row.max()
            cluster_centers = torch.linspace(row_min, row_max, self.n_clusters, device=weight.device).reshape(-1, 1)
            # K-Means 聚类
            for _ in range(self.max_iter):  # 迭代次数
                # 计算每个点到聚类中心的距离
                distances = torch.cdist(row.reshape(-1, 1), cluster_centers)
                # 分配每个点到最近的聚类中心
                labels = torch.argmin(distances, dim=1)
                # 更新聚类中心
                for k in range(self.n_clusters):
                    if (labels == k).sum() > 0:
                        cluster_centers[k] = row[labels == k].mean()
            self.row_cluster_centers.append(cluster_centers)
        return self.row_cluster_centers

    
    def quantize(self, weight):
        """
        对权重矩阵按行进行量化
        :param weight: 输入权重张量 (torch.Tensor)
        :return: 量化后的权重张量
        """
        print("cluster:", self.n_clusters)
        w_size = weight.size(-1)


        if self.group_size < 0:
            denominator = abs(self.group_size)
            assert w_size % denominator == 0, f"Weight size {w_size} must be divisible by {denominator}"
            group_size = w_size // denominator
        else:
            group_size = self.group_size
            #assert w_size % group_size == 0, f"Weight size {w_size} must be divisible by group_size {group_size}"

        org_shape = weight.shape
        
        weight=weight.reshape(-1,group_size)
        print("group_size:",group_size)
        print("cluster Quantizing weight with shape:", weight.shape)
        #raise NotImplementedError

        quantized_weight = torch.zeros_like(weight)
        for i, row in enumerate(weight):
            # 获取当前行的聚类中心
            cluster_centers = self.row_cluster_centers[i]
            # 计算每个点到聚类中心的距离
            distances = torch.cdist(row.reshape(-1, 1), cluster_centers)
            # 分配每个点到最近的聚类中心
            labels = torch.argmin(distances, dim=1)
            # 替换为聚类中心值
            quantized_row = cluster_centers[labels].squeeze()
            quantized_weight[i] = quantized_row

        quantized_weight = quantized_weight.reshape(org_shape)
        return quantized_weight

