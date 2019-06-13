from Cluster import Cluster
import operator

class DTree():

    """创建决策树"""
    def create_division_tree(self, data_set, labels):
        target_list = [example[-1] for example in data_set]

        # 所有分类相同时返回
        if target_list.count(target_list[0]) == len(target_list):
            return target_list[0]

        # 已经遍历完所有特征
        if len(data_set[0]) == 1:
            return self.get_top_class(target_list)

        # 选取最好的特征
        cluster = Cluster()
        best_feat = cluster.choose_best_feature_id3(data_set)
        best_feat_label = labels[best_feat]

        # 划分
        my_tree = {best_feat_label: {}}
        del (labels[best_feat])
        value_set = set([example[best_feat] for example in data_set])

        for value in value_set:
            sub_labels = labels[:]
            my_tree[best_feat_label][value] = self.create_division_tree(cluster.split_data_set(data_set, best_feat, value),
                                                                   sub_labels)

        return my_tree

    """从多个分类中选取出现频率最高的分类"""
    def get_top_class(self, labels):
        label_counts = {}
        for vote in labels:
            if vote not in label_counts.keys():
                label_counts[vote] = 0
            else:
                label_counts[vote] += 1
        sorted_label_count = sorted(label_counts.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_label_count[0][0]

    """遍历决策树对测试数据进行分类"""
    def classify(self, division_tree, feat_labels, test_vector):
        first_key = list(division_tree.keys())[0]
        second_dict = division_tree[first_key]

        feat_index = feat_labels.index(first_key)
        test_key = test_vector[feat_index]

        test_value = second_dict[test_key]
        if isinstance(test_value, dict):
            class_label = self.classify(test_value, feat_labels, test_vector)
        else:
            class_label = test_value
        return class_label


