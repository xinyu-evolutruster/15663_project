import os
import cv2
import time
import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as gaussian_filter

INF = 1e9

result_dir = '../result'
item_name = 'mobo2_tiny'


def create_grid_graph(height, width):
    G = nx.grid_2d_graph(height, width)
    return G


def compute_unary_term(image_stack, sigma=1.0):
    num_frames = image_stack.shape[2]

    unary_image_stack = np.zeros_like(image_stack)
    for i in range(num_frames):
        grad_x, grad_y = np.gradient(image_stack[:, :, i])
        grad_img = np.exp(-np.sqrt(grad_x ** 2 + grad_y ** 2))
        unary_image_stack[:, :, i] = gaussian_filter(grad_img, sigma=sigma, mode='constant', cval=0)

    return unary_image_stack


def get_unary_term(unary_image_stack, i, j, label):
    return unary_image_stack[i, j, label]


def get_pairwise_term(label1, label2, weight=0.001):
    return weight * np.abs(label1 - label2)


def compute_energy(image_stack, labels):
    return 0


def get_index(i, j, width):
    return i * width + j


def get_pos(index, width):
    i = index // width
    j = index - i * width
    return i, j


def alpha_expansion(image_stack, unary_image_stack, label_image, label):
    
    print("label = {}".format(label))

    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    G = nx.Graph()

    # Add the two terminals alpha and alpha_bar
    G.add_node('alpha')
    G.add_node('alpha_bar')

    height, width, _ = image_stack.shape

    start_time = time.time()

    for i in range(0, height):
        for j in range(0, width):

            # add the vertice
            idx = get_index(i, j, width)
            G.add_node(idx)

            # Add the edges connecting the pixels and the terminals
            G.add_edge('alpha', idx, capacity=get_unary_term(unary_image_stack, i, j, label))
            if label_image[i, j] == label:
                G.add_edge('alpha_bar', idx, capacity=INF)
            else:
                G.add_edge('alpha_bar', idx, capacity=get_unary_term(unary_image_stack, i, j, label_image[i, j]))

            # add the four edges connecting to the pixels
            for x, y in dirs:
                new_idx = get_index(i + x, j + y, width)
                if i + x >= 0 and j + y >= 0 and i + x < height and j + y < width:
                    # Create an auxiliary node if fp != fq
                    if label_image[i, j] != label_image[i + x, j + y]:
                        G.add_node((i, j))
                        G.add_edge(idx, (i, j), capacity=get_pairwise_term(label_image[i, j], label))
                        G.add_edge((i, j), new_idx, capacity=get_pairwise_term(label_image[i + x, j + y], label))
                        G.add_edge((i, j), 'alpha_bar', capacity=get_pairwise_term(label_image[i, j], label_image[i + x, j + y]))
                    else:
                        G.add_edge(idx, new_idx, capacity=get_pairwise_term(label_image[i, j], label))

    print("Finished creating graph, time: {}s".format(time.time() - start_time))

    start_time = time.time()

    source = 'alpha'
    terminal = 'alpha_bar'

    # cut_set = nx.minimum_edge_cut(G, source, terminal)
    cut_value, partition = nx.minimum_cut(G, source, terminal)
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, G[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    print("Finished computing minimum cut, time: {}s".format(time.time() - start_time))
    print("number of edges in the minimum cut: {}".format(len(cutset)))

    for edge in cutset:
        if edge[0] == 'alpha':
            index = edge[1]
            i, j = get_pos(index, width)
            # print("i = {}, j = {}".format(i, j))
            label_image[i, j] = label
        elif edge[1] == 'alpha':
            index = edge[0]
            i, j = get_pos(index, width)
            label_image[i, j] = label

    return label_image


def mrf_optimization(image_stack, color_img_stack=None, lambda_val=0.1, sigma=1.0, weight_pairwise=1.0, num_iterations=5):

    height, width, num_frames = image_stack.shape
    
    label_image = np.zeros((height, width), dtype=int)  # Initialize labels to zeros
    unary_image_stack = compute_unary_term(image_stack, sigma=1.0)
    
    energy = compute_energy(image_stack, labels=label_image)

    success = False
    for iteration in range(num_iterations):
        success = False

        # Create result directory
        res_dir = os.path.join(result_dir, item_name, 'iterations', '{}'.format(iteration))
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        for label in range(num_frames):
            label_image = alpha_expansion(image_stack, unary_image_stack, label_image, label)
            new_energy = compute_energy(image_stack, label_image)
            if new_energy < energy:
                success = True
                energy = new_energy

            fig=plt.figure()
            ax = plt.axes()

            im = ax.imshow(label_image, cmap='viridis')
            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
            plt.colorbar(im, cax=cax)
            plt.savefig(os.path.join(res_dir, 'label_{}.png'.format(label)), bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        if color_img_stack is not None:
            stitched_image = color_img_stack[np.arange(height)[:, None], np.arange(width), :, label_image]
            plt.imshow(stitched_image)
            cv2.imwrite(os.path.join(result_dir, item_name, 'stitched_images', 'stitched_img_{}.png'.format(iteration)), stitched_image)

        # if not success:
        #     break

        # test
        # if iteration > 1:
        #     break

    return label_image


if __name__ == '__main__':

    img_stack = []
    color_img_stack = []

    data_dir = os.path.join(result_dir, item_name, 'images')

    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)

        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        color_img_stack.append(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[0:, 0:] / 255
        img_stack.append(img)

    img_stack = np.array(img_stack)
    img_stack = np.transpose(img_stack, (1, 2, 0))

    color_img_stack = np.array(color_img_stack)
    color_img_stack = np.transpose(color_img_stack, (1, 2, 3, 0))

    print("img_stack shape: ", img_stack.shape)
    print("color_img_stack shape: ", color_img_stack.shape)

    optimized_labels = mrf_optimization(img_stack, color_img_stack)

    # Visualize the result
    print(optimized_labels)
    plt.imshow(optimized_labels, cmap='viridis')
    plt.colorbar()
    plt.show()