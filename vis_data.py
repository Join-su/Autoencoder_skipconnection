import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

import matplotlib.pyplot as plt

from subprocess import call



class Visdata():
    def __init__(self, graph_save_path, graph_name) :
        self.graph_save_path = graph_save_path
        self.graph_name = graph_name


    def vis_dataF(self, x_train_encoded, y_train, vis_dim, n_predict, n_train, build_anim, y, num):
        cmap = plt.get_cmap('rainbow', 16)


        # 3-dim vis: show one view, then compile animated .gif of many angled views
        if vis_dim == 3:
            # Simple static figure
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            p = ax.scatter3D(x_train_encoded[:, 0], x_train_encoded[:, 1], x_train_encoded[:, 2],
                             c=y_train[:n_predict], cmap=cmap, edgecolor='black')
            fig.colorbar(p, drawedges=True)
            # plt.show()

            # Build animation from many static figures
            if build_anim:
                angles = np.linspace(180, 360, 20)
                i = 0
                for angle in angles:
                    fig = plt.figure()
                    ax = plt.axes(projection='3d')
                    ax.view_init(10, angle)
                    p = ax.scatter3D(x_train_encoded[:, 0], x_train_encoded[:, 1], x_train_encoded[:, 2],
                                     c=y_train[:n_predict], cmap=cmap, edgecolor='black')
                    fig.colorbar(p, drawedges=True)
                    outfile = 'anim/3dplot_step_' + chr(i + 97) + '.png'
                    plt.savefig(outfile, dpi=96)
                    i += 1
                call(['convert', '-delay', '50', 'anim/3dplot*', 'anim/3dplot_anim_' + str(n_train) + '.gif'])

        # 2-dim vis: plot and colorbar.
        elif vis_dim == 2:
            print(y_train)
            plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1],
                        c=y_train[:n_predict], edgecolor='black', cmap=cmap)
            plt.colorbar(drawedges=True)
            # plt.show()
        if num == 1:
            plt.savefig(self.graph_save_path + self.graph_name)
        elif num == 2:
            plt.title('{} - ARI: {:.3f}'.format('KMeans', adjusted_rand_score(y[:n_predict], y_train[:n_predict])))
            plt.savefig(self.graph_save_path + self.graph_name + '_K-means')
        elif num == 3:
            plt.title('{} - ARI: {:.3f}'.format('Spectral', adjusted_rand_score(y[:n_predict], y_train[:n_predict])))
            plt.savefig(self.graph_save_path + self.graph_name + '_Spectral')
        plt.close()