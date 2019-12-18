'''
IOU(intersection over union)
计算相交比
'''

def cal_iou(a, b):
    '''
    计算相交比
    :param a: a（左下角x，左下角y, 右上角x, 右上角y）
    :param b: b（左下角x，左下角y, 右上角x, 右上角y）
    :return: 
    '''
    # a和b的机构应该是(x1,y1,x2,y2)，即（左下角x，左下角y, 右上角x, 右上角y）

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def intersection(ai, bi):
    '''
    计算相交面积
    :param ai: ai（左下角x，左下角y, 右上角x, 右上角y）
    :param bi: bi（左下角x，左下角y, 右上角x, 右上角y）
    :return:
    '''
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h

def union(au, bu, area_intersection):
    '''
    计算相并面积
    :param au: au（左下角x，左下角y, 右上角x, 右上角y）
    :param bu: bu（左下角x，左下角y, 右上角x, 右上角y）
    :param area_intersection: 相交面积
    :return:
    '''
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union