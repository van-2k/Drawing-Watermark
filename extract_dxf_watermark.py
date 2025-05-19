import ezdxf
import numpy as np
import argparse
from sklearn.cluster import KMeans

def is_horizontal(line, angle_tol=5):
    start = np.array(line.dxf.start)
    end = np.array(line.dxf.end)
    vec = end - start
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    return abs(angle) < angle_tol or abs(angle - 180) < angle_tol

def angle_between(a, b, c):
    # 计算∠abc的度数
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    return np.degrees(angle)

def point_on_segment(p, a, b, tol=1.0):
    # 判断点p是否在ab线段上（允许一定容差）
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    if t < 0 or t > 1:
        return False
    closest = a + t * ab
    return np.linalg.norm(closest - p) < tol

def extract_watermark_dxf(input_path, clusters, segments):
    doc = ezdxf.readfile(input_path)
    msp = doc.modelspace()

    # 1. 获取所有足够长的水平主线
    xs = []
    for e in msp:
        if e.dxftype() == 'LINE':
            xs.extend([e.dxf.start[0], e.dxf.end[0]])
    x_min, x_max = min(xs), max(xs)
    drawing_width = x_max - x_min

    lines = []
    lengths = []
    min_length = drawing_width * 0.05  # 只保留长度大于图纸宽度30%的主线
    for e in msp:
        if e.dxftype() == 'LINE' and is_horizontal(e):
            start = np.array(e.dxf.start)
            end = np.array(e.dxf.end)
            length = np.linalg.norm(end - start)
            if length > min_length:
                lines.append(e)
                lengths.append([length])
    if len(lines) < clusters:
        raise Exception(f"水平主线数量不足，无法聚类为{clusters}个簇")

    # 2. 聚类
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(lengths)
    labels = kmeans.labels_

    # 3. 每簇内选中位数主线
    main_lines = []
    for k in range(clusters):
        cluster_lines = []
        for idx, e in enumerate(lines):
            if labels[idx] == k:
                start = np.array(e.dxf.start)
                end = np.array(e.dxf.end)
                length = np.linalg.norm(end - start)
                cluster_lines.append((e, length))
        if cluster_lines:
            cluster_lines.sort(key=lambda x: x[1])
            mid_idx = len(cluster_lines) // 2
            main_line = cluster_lines[mid_idx][0]
            main_lines.append(main_line)

    # 4. 按长度降序排序，保证顺序一致
    main_lines = sorted(main_lines, key=lambda e: -np.linalg.norm(np.array(e.dxf.end) - np.array(e.dxf.start)))

    # 5. 获取WATERMARK图层上的所有短线
    watermark_lines = []
    for e in msp:
        if e.dxftype() == 'LINE' and e.dxf.layer == 'WATERMARK':
            start = np.array(e.dxf.start)
            end = np.array(e.dxf.end)
            watermark_lines.append((start, end))

    # 6. 对每条主线等分，依次提取比特
    bits = []
    for main_line in main_lines:
        start = np.array(main_line.dxf.start)
        end = np.array(main_line.dxf.end)
        vec = (end - start) / segments
        for i in range(segments):
            seg_start = start + i * vec
            seg_end = start + (i + 1) * vec
            center = (seg_start + seg_end) / 2
            radius = np.linalg.norm(seg_end[:2] - seg_start[:2]) / 2
            dir_vec = seg_end[:2] - seg_start[:2]
            dir_vec_unit = dir_vec / np.linalg.norm(dir_vec)
            norm_vec = np.array([-dir_vec[1], dir_vec[0]]) / np.linalg.norm(dir_vec)
            z = center[2]
            found = False
            for line_start, line_end in watermark_lines:
                mid = (line_start + line_end) / 2
                if abs(mid[2] - z) > 1.0:
                    continue
                if point_on_segment(mid[:2], seg_start[:2], seg_end[:2]):
                    P_xy = mid[:2]
                    PO = P_xy - center[:2]
                    a = 1
                    b = 2 * np.dot(norm_vec, PO)
                    c = np.dot(PO, PO) - radius**2
                    delta = b**2 - 4*a*c
                    if delta < 0:
                        continue
                    t1 = (-b + np.sqrt(delta)) / (2*a)
                    t2 = (-b - np.sqrt(delta)) / (2*a)
                    t = t1 if abs(t1) > abs(t2) else t2
                    Q_xy = P_xy + norm_vec * t
                    A = seg_start[:2]
                    B = seg_end[:2]
                    Q = Q_xy
                    alpha = angle_between(Q, A, B)
                    beta = angle_between(Q, B, A)
                    diff = abs(alpha - beta)
                    if 0 <= diff < 30:
                        bits.append('0')
                    elif 30 <= diff < 60:
                        bits.append('1')
                    else:
                        bits.append('?')
                    found = True
                    break
            if not found:
                bits.append('?')
    return ''.join(bits)

def main():
    parser = argparse.ArgumentParser(description="DXF水印提取（多主线聚类分段+短线角度差值法）")
    parser.add_argument('--input', type=str, required=True, help='输入DXF文件路径')
    parser.add_argument('--clusters', type=int, required=True, help='主线聚类数（主线数）')
    parser.add_argument('--segments', type=int, required=True, help='每条主线分段数')
    args = parser.parse_args()

    watermark = extract_watermark_dxf(args.input, args.clusters, args.segments)
    print(f"提取的水印比特串为: {watermark}")

if __name__ == "__main__":
    main() 