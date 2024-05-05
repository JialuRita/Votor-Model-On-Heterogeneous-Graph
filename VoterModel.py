import copy
import multiprocessing
import threading
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import random
from tqdm import tqdm

random.seed(3407)

# 生成2d格子网络
def create_network(width, height, p):
    G = nx.grid_2d_graph(width, height,
                         periodic=False)  # 如果设置periodic=True，则为环形
    nodes = list(G.nodes())
    for node in nodes:
        for target in nodes:
            if target != node and (
                    node, target) not in G.edges() and random.random() < p:
                G.add_edge(node, target)  #随机增加边
    return G

# 可视化节点状态变化
def visual(G, path):
    plt.figure()
    # 设置节点的位置
    pos = {(x, y): (y, -x) for x, y in G.nodes()}
    # 根据节点的 status 属性设置颜色
    node_colors = ['pink' if G.nodes[node]['status'] == 1 else 'royalblue' for node in
                   G.nodes()]
    # 绘制图像
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=5)
    plt.savefig(path)
    plt.close()
    # plt.show()


def initial(G, p):
    positive_nodes_num = round(len(G) * p)
    positive_nodes = random.sample(sorted(G.nodes()), positive_nodes_num)
    for node in G.nodes:
        G.nodes[node]['status'] = -1
    for node in positive_nodes:
        G.nodes[node]['status'] = 1
    nx.write_gexf(G, "./G.net")  # 保存网络为.net文件
    return G


def process(G):
    node = random.choice(list(G.nodes()))
    neighbors = list(G.neighbors(node))
    if neighbors:
        selected_neighbor = random.choice(neighbors)
        G.nodes[node]['status'] = G.nodes[selected_neighbor]['status']
    else:
        print("Node:", node, "has no neighbors.")
    return G


def cal_na(G):
    num = 0
    for edge in G.edges:
        if G.nodes[edge[0]]['status'] != G.nodes[edge[1]]['status']:
            num += 1
    return num / float(G.size())


def draw(dict_list, path, title, label_lis=None):
    marker_lis = ['o', 's', '<', '>', '*', '^']
    plt.figure(figsize=(10, 8))
    plt.tight_layout()
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号显示正常
    for i in range(len(dict_list)):
        dict = dict_list[i]
        keys = dict.keys()
        values = dict.values()
        if label_lis == None:
            plt.plot(keys, values, marker=marker_lis[i], color='royalblue')
        else:
            plt.plot(keys, values, marker=marker_lis[i], color='royalblue',
                     label=label_lis[i])
    plt.title(title)
    plt.xlabel('t')
    plt.ylabel('na')
    plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.savefig(path)
    plt.close()  # Close the figure to free up memory


def generate_video(img_folder, output_path):
    # 获取文件夹中所有jpg文件，按数字顺序排序
    file_names = sorted(
        [f for f in os.listdir(img_folder) if f.endswith('.jpg')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    # 检查是否有图片文件
    if not file_names:
        print("No '.jpg' images found in the folder.")
        return

    # 创建一个视频写入对象，指定输出文件和帧率
    with imageio.get_writer(output_path, fps=50) as writer:
        for filename in file_names:
            file_path = os.path.join(img_folder, filename)
            if os.path.exists(file_path):  # 确认文件存在
                img = imageio.imread(file_path)  # 使用imageio读取图片
                writer.append_data(img)  # 添加到视频
            else:
                print(f"File not found: {file_path}")
    print("Video created successfully at", output_path)


t_lis = [round(10 ** i) for i in np.linspace(0, 7, 50)]
t_lis = list(set(t_lis))
print(t_lis)

def part1(info: str):
    temp_list = copy.deepcopy(t_lis)
    G = create_network(100, 100, 0.0001)
    print("create end")
    initial(G, 0.5)
    visual(G, "./img\\0.jpg")
    dict = {}
    for t in tqdm(range(1, 10000001), desc=f"{info} Processing: "):
        G = process(G)
        if t % 1000 == 0:
            visual(G, f"./img\\{t}.jpg")
        if t in t_lis:
            na = cal_na(G)
            dict[t] = na
            print(f'{t} {na}')
            if na == 0:
                break
    draw([dict], "./part1.jpg", 'N = 10000 p = 0.0001')


def part2(info: str):
    temp_list = copy.deepcopy(t_lis)
    img_folder = "./img"
    output_path = "./task2.mp4"
    generate_video(img_folder, output_path)

    # part2
    # 变化N,p = 0.001
    dict_list = []
    N_lis = []
    for size in [10, 40, 100]:
        print("part2" + str(size))
        G = create_network(size, size, 0.0001)
        initial(G, 0.5)
        dict = {}
        for t in tqdm(range(1, 10000001), desc=f"{info} Processing: "):
            G = process(G)
            if t in temp_list:
                na = cal_na(G)
                dict[t] = na
                print(f'{t} {na}')
                if na == 0:
                    break
        dict_list.append(dict)
        N_lis.append(size * size)
    label_lis = [f'N = {N}' for N in N_lis]
    draw(dict_list, "./part2_N.jpg", 'p = 0.0001', label_lis)


def part3(info: str):
    temp_list = copy.deepcopy(t_lis)
    p_lis = [0, 0.0001, 0.0002, 0.0005, 0.001]
    dict_list = []
    for p in p_lis:
        print(p)
        G = create_network(100, 100, p)
        initial(G, 0.5)
        dict = {}
        for t in tqdm(range(1, 10000001), desc=f"{info} Processing: "):
            G = process(G)
            if t in temp_list:
                na = cal_na(G)
                dict[t] = na
                print(f'{t} {na}')
                if na == 0:
                    break
        dict_list.append(dict)
    label_lis = [f'p = {p}' for p in p_lis]
    draw(dict_list, "./part2_p.jpg", 'N = 10000', label_lis)


def part4(info: str):
    temp_list = copy.deepcopy(t_lis)
    dict_list = []
    N_lis = []
    for size in range(10, 101, 30):
        print(size)
        G = create_network(size, size, 0)
        initial(G, 0.5)
        dict = {}
        for t in tqdm(range(1, 10000001), desc=f"{info} Processing: "):
            G = process(G)
            if t in temp_list:
                na = cal_na(G)
                dict[t] = na
                print(f'{t} {na}')
                if na == 0:
                    break
        dict_list.append(dict)
        N_lis.append(size * size)
    label_lis = [f'N = {N}' for N in N_lis]
    draw(dict_list, "./part2_N_p=0.jpg", 'p = 0', label_lis)


def main():
    processes = []
    func1 = [part1, part2, part3, part4]
    info1 = ["part1", "part2", "part3", "part4"]
    for i in range(4):  # 创建4个进程
        p = multiprocessing.Process(target=func1[i], args=(info1[i],))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    for i in tqdm(range(1, 100)):
        time.sleep(0.01)
    main()