import ezdxf
import numpy as np
import argparse
from sklearn.cluster import KMeans


def angle_between(a, b, c):
    """计算∠abc的度数"""
    ba = a - b
    bc = c - b
    # 检查向量长度是否为0
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm == 0 or bc_norm == 0:
        return 0
    cos_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    return np.degrees(angle)


def get_line_angle(line):
    """获取线段与x轴的夹角（-180到180度）"""
    start = np.array(line.dxf.start)
    end = np.array(line.dxf.end)
    vec = end - start
    return np.degrees(np.arctan2(vec[1], vec[0]))


def get_dominant_angle(lines):
    """获取图纸的主要方向角度"""
    angles = []
    for line in lines:
        angle = get_line_angle(line)
        angles.append(angle)
    
    # 将角度转换到0-180度范围
    angles = [angle % 180 for angle in angles]
    
    # 使用直方图统计最频繁的角度
    hist, bins = np.histogram(angles, bins=180, range=(0, 180))
    dominant_angle = bins[np.argmax(hist)]
    
    return dominant_angle


def rotate_point(point, center, angle_degrees):
    """绕中心点旋转点"""
    angle_rad = np.radians(angle_degrees)
    x, y = point[0] - center[0], point[1] - center[1]
    cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)
    x_new = x * cos_theta - y * sin_theta
    y_new = x * sin_theta + y * cos_theta
    return np.array([x_new + center[0], y_new + center[1]])


def embed_watermark_dxf(input_path, output_path, watermark_bits, clusters=3, segments=8, search_steps=100):
    doc = ezdxf.readfile(input_path)
    msp = doc.modelspace()

    # 1. 计算图纸尺寸
    xs = []
    ys = []
    for e in msp:
        if e.dxftype() == 'LINE':
            xs.extend([e.dxf.start[0], e.dxf.end[0]])
            ys.extend([e.dxf.start[1], e.dxf.end[1]])
    
    if not xs or not ys:
        raise Exception("图纸中没有线段")
    
    drawing_width = max(xs) - min(xs)
    drawing_height = max(ys) - min(ys)
    drawing_size = max(drawing_width, drawing_height)
    
    # 设置线段最小长度阈值（使用图纸尺寸的百分比）
    line_min_length = drawing_size * 0.02  # 线段最小长度为图纸尺寸的2%

    # 2. 获取所有符合条件的线段
    lines = []
    lengths = []
    for e in msp:
        if e.dxftype() == 'LINE':
            start = np.array(e.dxf.start)
            end = np.array(e.dxf.end)
            # 只使用xy平面计算长度
            length = np.linalg.norm(end[:2] - start[:2])
            # 只选择长度大于阈值的线段
            if length >= line_min_length:
                lines.append(e)
                lengths.append([length])

    if len(lines) < clusters:
        raise Exception(f"线段数量不足，无法聚类为{clusters}个簇")

    # 3. 计算图纸中心点
    center = np.array([np.mean(xs), np.mean(ys)])

    # 4. 聚类选择线段
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(lengths)
    labels = kmeans.labels_

    # 5. 每簇内选中位数线段
    selected_lines = []
    for k in range(clusters):
        cluster_lines = []
        for idx, e in enumerate(lines):
            if labels[idx] == k:
                start = np.array(e.dxf.start)
                end = np.array(e.dxf.end)
                # 只使用xy平面计算长度
                length = np.linalg.norm(end[:2] - start[:2])
                cluster_lines.append((e, length))
        if cluster_lines:
            cluster_lines.sort(key=lambda x: x[1])
            mid_idx = len(cluster_lines) // 2
            mid_length = cluster_lines[mid_idx][1]
            # 找到所有长度等于中位数的线段
            candidates = [e for e, l in cluster_lines if np.isclose(l, mid_length)]
            # 选择 handle 最大的那一条
            selected_line = max(candidates, key=lambda e: int(e.dxf.handle, 16))
            selected_lines.append(selected_line)

    # 6. 按长度降序排序，保证顺序一致
    selected_lines = sorted(selected_lines, key=lambda e: -np.linalg.norm(np.array(e.dxf.end)[:2] - np.array(e.dxf.start)[:2]))

    print("\n本次嵌入水印的主线段如下：")
    for line in selected_lines:
        start = np.array(line.dxf.start)
        end = np.array(line.dxf.end)
        length = np.linalg.norm(end[:2] - start[:2])
        print(f"线段 handle: {line.dxf.handle}")
        print(f"  起点: ({start[0]:.3f}, {start[1]:.3f})")
        print(f"  终点: ({end[0]:.3f}, {end[1]:.3f})")
        print(f"  长度: {length:.3f}")

    # 7. 检查水印长度
    total_bits = clusters * segments
    if len(watermark_bits) != total_bits:
        raise Exception(f"水印比特串长度应为{total_bits}，实际为{len(watermark_bits)}")

    # 8. 对每条线段等分嵌入
    bit_idx = 0
    fixed_length = 0.05  # 短线总长度
    short_line_half_len = fixed_length / 2  # 短线半长

    for line in selected_lines:
        start = np.array(line.dxf.start)
        end = np.array(line.dxf.end)
        # 保存z坐标
        z = start[2] if len(start) > 2 else 0
        
        # 只使用xy平面进行计算
        start_xy = start[:2]
        end_xy = end[:2]
        
        vec = (end_xy - start_xy) / segments
        for i in range(segments):
            seg_start = start_xy + i * vec
            seg_end = start_xy + (i + 1) * vec
            center_seg = (seg_start + seg_end) / 2
            radius = np.linalg.norm(seg_end - seg_start) / 2
            dir_vec = seg_end - seg_start
            dir_vec_unit = dir_vec / np.linalg.norm(dir_vec)
            norm_vec = np.array([-dir_vec[1], dir_vec[0]]) / np.linalg.norm(dir_vec)
            
            bit = watermark_bits[bit_idx]
            found = False
            for offset in np.linspace(-radius, radius, search_steps):
                P_xy = center_seg + dir_vec_unit * offset
                PO = P_xy - center_seg
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
                A = seg_start
                B = seg_end
                Q = Q_xy
                alpha = angle_between(Q, A, B)
                beta = angle_between(Q, B, A)
                diff = abs(alpha - beta)
                if (bit == '0' and 0 <= diff < 30) or (bit == '1' and 30 <= diff < 60):
                    perp_vec = norm_vec
                    line_start = P_xy - perp_vec * short_line_half_len
                    line_end = P_xy + perp_vec * short_line_half_len
                    msp.add_line((line_start[0], line_start[1], z), 
                               (line_end[0], line_end[1], z), 
                               dxfattribs={'layer': 'WATERMARK'})
                    found = True
                    break
            if not found:
                print(f"警告：线段{line.dxf.handle} 第{i+1}段未找到满足条件的嵌入点，跳过")
            bit_idx += 1

    doc.saveas(output_path)
    print(f"水印嵌入完成，结果已保存到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="自适应聚类水印嵌入（画圆+角度差值法）")
    parser.add_argument('--input', type=str, required=True, help='输入DXF文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出带水印DXF文件路径')
    parser.add_argument('--watermark', type=str, required=True, help='水印比特串，如10101010...')
    parser.add_argument('--clusters', type=int, default=3, help='线段聚类数')
    parser.add_argument('--segments', type=int, default=8, help='每条线段分段数')
    args = parser.parse_args()

    embed_watermark_dxf(args.input, args.output, args.watermark, args.clusters, args.segments)

if __name__ == "__main__":
    main() 