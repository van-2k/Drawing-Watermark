import ezdxf
import numpy as np
import argparse
from math import cos, sin, radians

def rotate_point(point, center, angle_degrees):
    """绕中心点旋转点"""
    angle_rad = radians(angle_degrees)
    x, y = point[0] - center[0], point[1] - center[1]
    cos_theta, sin_theta = cos(angle_rad), sin(angle_rad)
    x_new = x * cos_theta - y * sin_theta
    y_new = x * sin_theta + y * cos_theta
    return (x_new + center[0], y_new + center[1])

def rotate_dxf(input_path, output_path, angle_degrees):
    """旋转DXF文件"""
    # 读取DXF文件
    doc = ezdxf.readfile(input_path)
    msp = doc.modelspace()

    # 计算图纸中心点
    xs = []
    ys = []
    for e in msp:
        if e.dxftype() == 'LINE':
            xs.extend([e.dxf.start[0], e.dxf.end[0]])
            ys.extend([e.dxf.start[1], e.dxf.end[1]])
    
    if not xs or not ys:
        raise Exception("图纸中没有线段")
    
    center_x = (max(xs) + min(xs)) / 2
    center_y = (max(ys) + min(ys)) / 2
    center = (center_x, center_y)
    
    print(f"图纸中心点: ({center_x:.3f}, {center_y:.3f})")
    print(f"旋转角度: {angle_degrees}度")

    # 旋转所有线段
    for e in msp:
        if e.dxftype() == 'LINE':
            # 旋转起点
            start = rotate_point(e.dxf.start, center, angle_degrees)
            # 旋转终点
            end = rotate_point(e.dxf.end, center, angle_degrees)
            # 更新线段
            e.dxf.start = start
            e.dxf.end = end

    # 保存旋转后的DXF文件
    doc.saveas(output_path)
    print(f"已保存旋转后的DXF文件: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="旋转DXF工程图文件")
    parser.add_argument('--input', type=str, required=True, help='输入DXF文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出DXF文件路径')
    parser.add_argument('--angle', type=float, required=True, help='旋转角度（度）')
    args = parser.parse_args()

    rotate_dxf(args.input, args.output, args.angle)

if __name__ == "__main__":
    main() 