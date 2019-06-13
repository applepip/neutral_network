from math import log

class Cluster():


    def calc_entropy(self, data_set):
        """计算数据集的熵"""
        count = len(data_set)
        tar_counts = {}

        # 统计数据集中每种分类的个数
        for row in data_set:
            tmp_tar = row[-1]
            if tmp_tar not in tar_counts.keys():
                tar_counts[tmp_tar] = 1
            else:
                tar_counts[tmp_tar] += 1

        # 计算熵
        entropy = 0.0
        for key in tar_counts:
            prob = float(tar_counts[key]) / count
            entropy -= prob * log(prob, 2)

        return entropy


    def split_data_set(self, data_set, axis, value):
        """根据指定条件分割数据集"""
        # 划分后的新数据集
        new_data_set = []

        for row in data_set:
            if row[axis] == value:
                split_vector = row[:axis]
                split_vector.extend(row[axis + 1:])
                new_data_set.append(split_vector)

        return new_data_set



    """选取信息增益最高的特征"""
    def choose_best_feature_id3(self, data_set):

        # 特征总数
        feature_count = len(data_set[0]) - 1

        # 数据集的原始熵
        base_entropy = self.calc_entropy(data_set)

        # 最大的信息增益
        best_gain = 0.0
        # 信息增益最大的特征
        best_feature = -1

        # 遍历计算每个特征
        for i in range(feature_count):
            feature = [example[i] for example in data_set]
            #同一特征下的样本值分类
            feature_value_set = set(feature)
            new_entropy = 0.0

            # 计算信息增益
            for value in feature_value_set:
                sub_data_set = self.split_data_set(data_set, i, value)
                prob = len(sub_data_set) / float(len(data_set))
                new_entropy += prob * self.calc_entropy(sub_data_set)

            gain = base_entropy - new_entropy

            # 比较得出最大的信息增益
            if gain > best_gain:
                best_gain = gain
                best_feature = i

        return best_feature

    """根据增益率选取划分特征"""
    def choose_best_feature_c45(self, data_set):
        # 特征总数
        feature_count = len(data_set[0]) - 1

        # 数据集的原始熵
        base_entropy = self.calc_entropy(data_set)

        # 最大的信息增益率
        best_gain_ratio = 0.0
        # 信息增益率最大的特征
        best_feature = -1

        # 遍历计算每个特征
        for i in range(feature_count):
            feature = [example[i] for example in data_set]
            feature_value_set = set(feature)

            new_entropy = 0.0
            intrinsic_value = 0.0

            # 计算信息增益
            for value in feature_value_set:
                sub_data_set = self.split_data_set(data_set, i, value)
                prob = len(sub_data_set) / float(len(data_set))
                new_entropy += prob * self.calc_entropy(sub_data_set)
                intrinsic_value -= prob * log(prob, 2)

            gain = base_entropy - new_entropy
            gain_ratio = gain / intrinsic_value

            # 比较得出最大的信息增益率
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_feature = i

        return best_feature


    """计算数据集的基尼值"""
    def calc_gini(self, data_set):
        count = len(data_set)
        label_counts = {}

        # 统计数据集中每种分类的个数
        for row in data_set:
            label = row[-1]
            if label not in label_counts.keys():
                label_counts[label] = 1
            else:
                label_counts[label] += 1

        # 计算基尼值
        gini = 1.0
        for key in label_counts:
            prob = float(label_counts[key]) / count
            gini -= prob * prob

        return gini

    """根据基尼指数选择划分特征"""
    def choose_best_feature_gini_idx(self, data_set):
        """根据基尼指数选择划分特征"""
        feature_count = len(data_set[0]) - 1
        # 最小基尼指数
        min_gini_index = 0.0
        # 基尼指数最小的特征
        best_feature = -1

        # 遍历计算每个特征
        for i in range(feature_count):
            feature = [example[i] for example in data_set]
            feature_value_set = set(feature)

            # 基尼指数
            gini_index = 0.0

            # 计算基尼指数
            for value in feature_value_set:
                sub_data_set = self.split_data_set(data_set, i, value)
                prob = len(sub_data_set) / float(len(data_set))
                gini_index += prob * self.calc_gini(sub_data_set)


            # 比较得出最小的基尼指数
            if gini_index < min_gini_index or gini_index == 0.0:
                min_gini_index = gini_index
                best_feature = i

        return best_feature






