import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
from igraph import Graph

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel

n_components = 5
gamma = 50


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    img = np.asarray(img, dtype=np.float64)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect
    w -= x
    h -= y

    # Initalize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    prev_energy = 0
    num_iters = 1000
    for i in range(num_iters):

        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        relative_energy_change = abs(energy - prev_energy) / energy

        if check_convergence(relative_energy_change):
            break

        prev_energy = energy

    mask[mask == GC_PR_BGD] = GC_BGD
    mask[mask == GC_PR_FGD] = GC_FGD

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask):
    bg = img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)]
    fg = img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)]

    kmeans_bg = KMeans(n_clusters=n_components, random_state=0, n_init="auto").fit(bg)
    kmeans_fg = KMeans(n_clusters=n_components, random_state=0, n_init="auto").fit(fg)

    bgGMM = calc_components(bg, kmeans_bg.labels_)
    fgGMM = calc_components(fg, kmeans_fg.labels_)

    return bgGMM, fgGMM


def calc_components(pixels_arr, component_index):
    gmm = []

    for i in range(n_components):
        sigma_i, mu_i = cv2.calcCovarMatrix(pixels_arr[component_index == i], None,
                                            cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
        det_i = np.linalg.det(sigma_i)
        inverse_i = np.linalg.inv(sigma_i)
        pi_i = pixels_arr[component_index == i].size / pixels_arr.size

        component_i = [mu_i[0], inverse_i, det_i, pi_i]
        gmm.append(component_i)

    return gmm


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    bg = img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)]
    fg = img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)]

    bg_component_index = calc_component_index(bg, bgGMM)
    fg_component_index = calc_component_index(fg, fgGMM)

    bgGMM = calc_components(bg, bg_component_index)
    fgGMM = calc_components(fg, fg_component_index)

    return bgGMM, fgGMM


def calc_component_index(pixel_arr, gmm):
    mu_arr = np.array([gmm[i][0] for i in range(len(gmm))])

    distances_from_mu = np.column_stack([np.sum((pixel_arr - mu_arr[i]) ** 2, axis=1) for i in range(len(mu_arr))])

    return np.argmin(distances_from_mu, axis=1)


def calculate_mincut(img, mask, bgGMM, fgGMM):
    img_size = img.shape[0] * img.shape[1]

    g = Graph(img_size + 2, directed=False)
    source = img_size  # Background Node
    target = img_size + 1  # Foreground Node

    two_dim_img = img.reshape(img_size, img.shape[2])
    pixels_id = np.arange(img_size).reshape(img.shape[0], img.shape[1])
    two_dim_mask = mask.reshape(-1)

    nlinks_edges, nlinks_capacities, beta = calculate_nlinks(img, two_dim_img, pixels_id)

    K = 8 * np.max(nlinks_capacities)
    tlinks_edges, tlinks_capacities = calculate_tlinks(two_dim_img, two_dim_mask, bgGMM, fgGMM, K)

    edges = np.concatenate((nlinks_edges, tlinks_edges)).tolist()
    capacities = np.concatenate((nlinks_capacities, tlinks_capacities)).tolist()

    g.add_edges(edges)
    g_min_cut = g.mincut(source, target, capacities)

    bg = g_min_cut.partition[0]
    fg = g_min_cut.partition[1]
    bg.remove(source)
    fg.remove(target)
    min_cut = [bg, fg]

    # calculating energy as described in the original paper
    # does not yield good results when using float64 for img, but decent results for int8.
    # also gives strange results for the image "fullmoon"
    # energy = calc_energy(img, mask, two_dim_mask, beta, bgGMM, fgGMM)

    # using energy from the mincut
    energy = g_min_cut.value

    return min_cut, energy


def calculate_nlinks(img, two_dim_img, pixels_id, cache={}):
    if len(cache) == 0:  # using memoization to optimize the calculation: computing the n-links only once
        beta = calc_beta(img)

        def N_adjacent(n, m): return gamma * np.exp(-beta * np.linalg.norm(two_dim_img[n] - two_dim_img[m]) ** 2)

        def N_diagonal(n, m): return (gamma / 2 ** 0.5) * np.exp(
            -beta * np.linalg.norm(two_dim_img[n] - two_dim_img[m]) ** 2)

        top_left_edges = np.column_stack((pixels_id[1:, 1:].reshape(-1), pixels_id[:-1, :-1].reshape(-1)))
        top_left_capacities = [N_diagonal(n, m) for n, m in top_left_edges]

        top_edges = np.column_stack((pixels_id[1:, :].reshape(-1), pixels_id[:-1, :].reshape(-1)))
        top_capacities = [N_adjacent(n, m) for n, m in top_edges]

        top_right = np.column_stack((pixels_id[1:, :-1].reshape(-1), pixels_id[:-1, 1:].reshape(-1)))
        top_right_capacities = [N_diagonal(n, m) for n, m in top_right]

        left_edges = np.column_stack((pixels_id[:, 1:].reshape(-1), pixels_id[:, :-1].reshape(-1)))
        left_capacities = [N_adjacent(n, m) for n, m in left_edges]

        nlinks_edges = np.concatenate((top_left_edges, top_edges, top_right, left_edges))
        nlinks_capacities = np.concatenate((top_left_capacities, top_capacities, top_right_capacities, left_capacities))

        cache[0] = (nlinks_edges, nlinks_capacities, beta)

    return cache[0]


def calc_beta(img):
    top_left_dist = np.sum((img[1:, 1:] - img[:-1, :-1]) ** 2)  # sum of img_size - img.shape[0] - img.shape[1] elements
    top_dist = np.sum((img[1:, :] - img[:-1, :]) ** 2)  # sum of img_size - img.shape[0] elements
    top_right_dist = np.sum(
        (img[1:, :-1] - img[:-1, 1:]) ** 2)  # sum of img_size - img.shape[0] - img.shape[1] elements
    left_dist = np.sum((img[:, 1:] - img[:, :-1]) ** 2)  # sum of img_size - img.shape[1] elements

    img_size = img.shape[0] * img.shape[1]
    total_elements_summed = 4 * img_size - 3 * (img.shape[0] + img.shape[1])
    expected_dist = (left_dist + top_left_dist + top_dist + top_right_dist) / total_elements_summed

    beta = 1 / (2 * expected_dist)
    return beta


def calculate_tlinks(two_dim_img, two_dim_mask, bgGMM, fgGMM, K):
    bg_indecies = np.where(two_dim_mask == GC_BGD)[0]
    fg_indecies = np.where(two_dim_mask == GC_FGD)[0]
    pr_bg_indecies = np.where(two_dim_mask == GC_PR_BGD)[0]
    pr_fg_indecies = np.where(two_dim_mask == GC_PR_FGD)[0]

    source = len(two_dim_img)  # Background Node
    target = len(two_dim_img) + 1  # Foreground Node

    def D_fore_back(edges, gmm):
        indexes_connected_to_st = np.array(edges)[:, 1]  # st is shortcut for source or target
        p_to_st = two_dim_img[indexes_connected_to_st]  # z values of pixels connected to source of target

        # equation (2) from [Justin F Talbot and Xiaoqian Xu]
        D = -np.log(np.sum([gmm[i][3] / np.sqrt(gmm[i][2]) * np.exp(
            -0.5 * np.sum(np.dot(p_to_st - gmm[i][0], gmm[i][1]) * (p_to_st - gmm[i][0]), axis=1)) for i in
                            range(n_components)], axis=0))
        return D

    source_to_bg_edges = np.column_stack(([source] * len(bg_indecies), bg_indecies))
    source_to_bg_cap = [K] * len(source_to_bg_edges)

    source_to_pr_fg_edges = np.column_stack(([source] * len(pr_fg_indecies), pr_fg_indecies))
    source_to_pr_fg_cap = D_fore_back(source_to_pr_fg_edges, fgGMM)

    target_to_fg_edges = np.column_stack(([target] * len(fg_indecies), fg_indecies))
    target_to_fg_cap = [K] * len(target_to_fg_edges)

    target_to_pr_fg_edges = np.column_stack(([target] * len(pr_fg_indecies), pr_fg_indecies))
    target_to_pr_fg_cap = D_fore_back(target_to_pr_fg_edges, bgGMM)

    tlinks_edges = np.concatenate(
        [source_to_bg_edges, source_to_pr_fg_edges, target_to_fg_edges, target_to_pr_fg_edges])
    tlinks_capacities = np.concatenate([source_to_bg_cap, source_to_pr_fg_cap, target_to_fg_cap, target_to_pr_fg_cap])

    if len(pr_bg_indecies) != 0:
        source_to_pr_bg_edges = np.column_stack(([source] * len(pr_bg_indecies), pr_bg_indecies))
        source_to_pr_bg_cap = D_fore_back(source_to_pr_bg_edges, fgGMM)

        target_to_pr_bg_edges = np.column_stack(([target] * len(pr_bg_indecies), pr_bg_indecies))
        target_to_pr_bg_cap = D_fore_back(target_to_pr_bg_edges, bgGMM)

        tlinks_edges = np.concatenate((tlinks_edges, source_to_pr_bg_edges, target_to_pr_bg_edges))
        tlinks_capacities = np.concatenate((tlinks_capacities, source_to_pr_bg_cap, target_to_pr_bg_cap))

    return tlinks_edges, tlinks_capacities


def calc_energy(img, mask, two_dim_mask, beta, bgGMM, fgGMM):
    U = calc_U(img, mask, bgGMM, fgGMM)
    V = calc_V(img, two_dim_mask, beta)

    # print(f"U: {U}, V: {V}")
    return U + V


def calc_U(img, mask, bgGMM, fgGMM):
    bg = img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)]
    fg = img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)]

    bg_component_index = calc_component_index(bg, bgGMM)
    fg_component_index = calc_component_index(fg, fgGMM)

    def calc_D(gmm, component_index, z_arr):
        components_mu = np.array([gmm[i][0] for i in range(len(gmm))])
        components_inv = np.array([gmm[i][1] for i in range(len(gmm))])
        components_det = np.array([gmm[i][2] for i in range(len(gmm))])
        components_pi = np.array([gmm[i][3] for i in range(len(gmm))])

        mu_arr = components_mu[component_index]
        det_arr = components_det[component_index]
        pi_arr = components_pi[component_index]

        dist = z_arr - mu_arr
        mult = dist.copy()
        for i in range(n_components):
            mult[component_index == i] = mult[component_index == i].dot(components_inv[i])

        mult = np.sum(mult * dist, axis=1)

        D = -np.log(pi_arr) + 0.5 * np.log(det_arr) + 0.5 * mult  # equation 9 in the original paper
        return np.sum(D)

    D_bg = calc_D(bgGMM, bg_component_index, bg)
    D_fg = calc_D(fgGMM, fg_component_index, fg)
    U = D_bg + D_fg
    return U


def calc_V(img, two_dim_mask, beta):
    fg_indecies = np.where(two_dim_mask == GC_PR_FGD)[0]
    bg_indecies = np.concatenate((np.where(two_dim_mask == GC_BGD)[0], np.where(two_dim_mask == GC_PR_BGD)[0]))

    in_bg = np.zeros((img.shape[0], img.shape[1]))
    np.put(in_bg, bg_indecies, 1)
    in_fg = np.zeros((img.shape[0], img.shape[1]))
    np.put(in_fg, fg_indecies, 1)

    top_left_dist = np.sum(img[1:, 1:] - img[:-1, :-1] ** 2, axis=2)
    top_dist = np.sum(img[1:, :] - img[:-1, :] ** 2, axis=2)
    top_right_dist = np.sum(img[1:, :-1] - img[:-1, 1:] ** 2, axis=2)
    left_dist = np.sum((img[:, 1:] - img[:, :-1]) ** 2, axis=2)

    # equation 11 in the original paper
    top_left_V = (in_fg[1:, 1:] == in_bg[:-1, :-1]) * np.exp(-beta * top_left_dist)
    top_V = (in_fg[1:, :] == in_bg[:-1, :]) * np.exp(-beta * top_dist)
    top_right_V = (in_fg[1:, :-1] == in_bg[:-1, 1:]) * np.exp(-beta * top_right_dist)
    left_V = (in_fg[:, 1:] == in_bg[:, :-1]) * np.exp(-beta * left_dist)

    # equation 11 in the original paper
    V = gamma * (np.sum(top_left_V) + np.sum(top_V) + np.sum(top_right_V) + np.sum(left_V))
    return V


def update_mask(mincut_sets, mask):
    two_dim_mask = mask.reshape(-1)

    new_bg = two_dim_mask[mincut_sets[0]]
    new_bg[new_bg != GC_BGD] = GC_PR_BGD

    new_fg = two_dim_mask[mincut_sets[1]]
    new_fg[new_fg != GC_FGD] = GC_PR_FGD

    two_dim_mask[mincut_sets[0]] = new_bg
    two_dim_mask[mincut_sets[1]] = new_fg

    mask = two_dim_mask.reshape(mask.shape[0], mask.shape[1])

    return mask


def check_convergence(energy):
    threshold = 0.0025
    convergence = False

    if energy <= threshold:
        convergence = True

    return convergence


def cal_metric(predicted_mask, gt_mask):
    fg_predicted = predicted_mask[predicted_mask == 1]
    fg_gt = gt_mask[gt_mask == 1]

    accuracy = np.sum(predicted_mask == gt_mask) / gt_mask.size

    cap = gt_mask[(gt_mask == 1) & (predicted_mask == 1)]
    jaccard = cap.size / (fg_predicted.size + fg_gt.size - cap.size)

    return accuracy, jaccard


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))

    img = cv2.imread(input_path)
    # low_blur_img = cv2.GaussianBlur(img, (7, 7), 0)
    # high_blur_img = cv2.GaussianBlur(img, (51, 51), 0)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
