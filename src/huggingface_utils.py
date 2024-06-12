import numpy as np
import os
import multiprocessing
import time
import torch
import matplotlib.pyplot as plt
# import scienceplots
# plt.style.use(['science', 'no-latex'])

class HuggingFaceUtils:
    """Collection of tools to perform spectral analysis and SVD on Transformer parameters"""
    
    def __init__(self, model):
        """
        Here, "model" should be a PyTorch HuggingFace model. This step exists to keep parameters on the CPU
        for multiprocessing over all available cores.
        """
        self.layer_list = [(name, param.to(torch.device('cpu'))) for name,param in model.named_parameters()]

    def block_diagonalize(self, A):
        """Take skew symmetric matrix and put in QSQ^T as presented in https://en.wikipedia.org/wiki/Skew-symmetric_matrix"""
        # A must be a antisymmetric (or skew symmetric) matrix
        assert np.array_equal(A, -A.T)

        d = A.shape[0]

        eigenvalues, eigenvectors = np.linalg.eig(A)
        imageigenvalues = np.imag(eigenvalues)
        realeigenvectors=np.real(eigenvectors)
        imageigenvectors=np.imag(eigenvectors)

        block_diag = np.zeros((d,d))
        V = []

        if d%2==0: # even dimention is an easy case
            for i in range(int(d/2)):
                block_diag[2*i,2*i+1]= imageigenvalues[2*i]
                block_diag[2*i+1,2*i]= imageigenvalues[2*i+1]
                V.append(realeigenvectors[:,2*i]*np.sqrt(2)) # scale V to be orthonormal, though not sure why it is not without scaling...
                V.append(imageigenvectors[:,2*i]*np.sqrt(2))
        else: # if odd dimention, there is a 0 eigen value and vector.
            find0 = 0
            for i in range(int((d-1)/2)):
                # need to skip 0 dimension
                if imageigenvalues[2*i]==0:
                    block_diag[2*i,2*i] = 0
                    V.append(realeigenvectors[:,2*i])
                    find0 = 1
                block_diag[2*i+find0,2*i+1+find0]= imageigenvalues[2*i+find0]
                block_diag[2*i+1+find0,2*i+find0]= imageigenvalues[2*i+1+find0]
                V.append(realeigenvectors[:,2*i+find0]*np.sqrt(2))
                V.append(imageigenvectors[:,2*i+find0]*np.sqrt(2))
            if imageigenvalues[d-1]==0: # check bounary condition
                block_diag[d-1,d-1] = 0
                V.append(realeigenvectors[:,d-1])

        V = np.array(V)
        V = V.T

        # A = V @ block_diag @ V.T
        return block_diag, V, imageigenvalues
    
    def svd_angles(self, M):
        """Computes the SVD of a matrix and the angles between the left and right singular vectors"""
        U, S, Vt = np.linalg.svd(M)

        # Compute the angles between left and right singular vectors
        angles = []
        for i in range(min(M.shape)): # assumes 2D
            u = U[:, i]
            v = Vt[i, :]
            angle = np.arccos(np.dot(u, v) / (np.linalg.norm(u)*np.linalg.norm(v)))
            angles.append(angle)

        return U, S, Vt, np.array(angles)
    
    def plot_single_mha_spectrum(self, Q, K, plot_path, layer_name, n_heads, max_vals):
        """
        Takes the eigenspectrum of the QK^T symmetric and antisymmetric matrices 
        as well as the SVD for one multi-head attention layer.
        """
        plot_file = '{}/{}.png'.format(plot_path, layer_name)
        npy_file = '{}/{}.npy'.format(plot_path, layer_name)

        fig, axs = plt.subplots(n_heads, 5, figsize=(60, n_heads*10))
        fig.tight_layout(pad=10.0)

        # Stores the matrices for all heads
        M = np.zeros((Q.shape[0], Q.shape[1], Q.shape[0]))
        M_sym = np.zeros_like(M)
        M_asym = np.zeros_like(M)
        
        label_font_size = 40
        tick_font_size = 40
        legend_font_size= 30

        info_arr = np.zeros((n_heads, 4, max_vals))
        # Compute for each head in a multihead attention layer
        for head in range(n_heads):
            M[:,head,:] = np.matmul(Q[:,head,:], K[:,head,:].T)
            M_sym[:,head,:] = (M[:,head,:] + M[:,head,:].T) / 2
            M_asym[:,head,:] = (M[:,head,:] - M[:,head,:].T) / 2

            *_, im_eig = self.block_diagonalize(M_asym[:,head,:])
            eig_val, eig_vec = np.linalg.eig(M_sym[:,head,:])
            U, S, Vt, angles = self.svd_angles(M[:,head,:])
            
            info_arr[head,0] = np.abs(im_eig[:max_vals])
            info_arr[head,1] = np.sort(eig_val)[::-1][:max_vals]
            info_arr[head,2] = S[:max_vals]
            info_arr[head,3] = angles[:max_vals]

            axs[head,0].scatter(np.arange(max_vals), np.abs(im_eig[:max_vals]), marker='o', linewidth=2,  color='red')
            axs[head,0].set_title(r"Antisymmetric $QK^T$ Spectrum for Head: {}".format(head+1), fontsize=30)
            axs[head,0].set_xlabel("First {} Eigenvalues (Descending)".format(max_vals), fontsize=label_font_size)
            axs[head,0].tick_params(axis='x', labelsize=tick_font_size)
            axs[head,0].tick_params(axis='y', labelsize=tick_font_size)
            axs[head,0].grid(color = 'gray',which='major', alpha=0.5)
            axs[head,0].grid(color = 'gray',which='minor', alpha=0.2)

            axs[head,1].scatter(np.arange(max_vals), np.sort(eig_val)[::-1][:max_vals], marker='o', linewidth=2, color='blue')
            axs[head,1].set_title(r"Symmetric $QK^T$ Spectrum for Head: {}".format(head+1), fontsize=30)
            axs[head,1].set_xlabel("First {} Eigenvalues (Descending)".format(max_vals), fontsize=label_font_size)
            axs[head,1].tick_params(axis='x', labelsize=tick_font_size)
            axs[head,1].tick_params(axis='y', labelsize=tick_font_size)
            axs[head,1].grid(color = 'gray',which='major', alpha=0.5)
            axs[head,1].grid(color = 'gray',which='minor', alpha=0.2)

            axs[head,2].scatter(np.arange(max_vals), S[:max_vals],  marker='o', linewidth=2, color='black')
            axs[head,2].set_title(r"Singular values of $QK^T$ for Head: {}".format(head+1), fontsize=30)
            axs[head,2].set_xlabel("First {} Singular Values (Descending)".format(max_vals), fontsize=label_font_size)
            axs[head,2].tick_params(axis='x', labelsize=tick_font_size)
            axs[head,2].tick_params(axis='y', labelsize=tick_font_size)
            axs[head,2].grid(color = 'gray',which='major', alpha=0.5)
            axs[head,2].grid(color = 'gray',which='minor', alpha=0.2)

            axs[head,3].scatter(np.arange(max_vals), angles[:max_vals],  marker='o', linewidth=2, color='magenta')
            axs[head,3].set_xlabel("First {} Singular Values (Descending)".format(max_vals), fontsize=label_font_size)
            axs[head,3].set_ylabel(r"$\theta$ (rad)", fontsize=label_font_size)
            axs[head,3].set_ylim([0, np.pi])
            axs[head,3].tick_params(axis='x', labelsize=tick_font_size)
            axs[head,3].tick_params(axis='y', labelsize=tick_font_size)
            axs[head,3].axhline(np.pi/2, color='black', linestyle='--', label=r"y=$\pi$/2 $\pm$ 95% conf.")  # Add the dashed line at y = π/2
            axs[head, 3].axhspan(np.pi/2 - 0.123, np.pi/2 + 0.123, facecolor='lightblue', alpha=0.5)
            axs[head,3].grid(color = 'gray', which='major', alpha=0.5)
            axs[head,3].grid(color = 'gray',which='minor', alpha=0.2)
            axs[head,3].legend(fontsize=legend_font_size)

            axs[head,4].scatter(np.arange(max_vals), np.abs(im_eig[:max_vals]), label=r"$M_{asym}$ eigenvalues",  marker='o', linewidth=2, color='red')
            axs[head,4].scatter(np.arange(max_vals), np.sort(eig_val)[::-1][:max_vals], label=r"$M_{sym}$ eigenvalues", marker='o', linewidth=2, color='blue')
            axs[head,4].scatter(np.arange(max_vals), S[:max_vals], label='Singluar Values M', marker='o', linewidth=2, color='black')
            axs[head,4].set_xlabel("First {} Values (Descending)".format(max_vals), fontsize=label_font_size)
            axs[head,4].set_ylabel("Magnitude", fontsize=label_font_size)

            axs[head,4].tick_params(axis='x', labelsize=tick_font_size)
            axs[head,4].tick_params(axis='y', labelsize=tick_font_size)
            axs[head,4].grid(color = 'gray',which='major', alpha=0.5)
            axs[head,4].grid(color = 'gray',which='minor', alpha=0.2)
            axs[head,4].legend(fontsize=legend_font_size)
       
        plt.savefig(plot_file)
        np.save(npy_file, info_arr)
        plt.clf()
        plt.close()
    
    def plot_spectrum(self, plot_path, n_heads=12, max_vals=40):
        """Multiprocessing enabled spectral analysis over all multi-head attention layers in a network"""

        self.QK_layer_list = self.get_QK(n_heads)
        for *_, layer in self.QK_layer_list:
            plot_file = '{}/{}_spectrum.png'.format(plot_path, layer)
            os.makedirs(os.path.dirname(plot_file), exist_ok=True)
            
        args = [(Q, K, plot_path, layer_name, n_heads, max_vals) for (Q, K, layer_name) in self.QK_layer_list]
        st = time.time()
        with multiprocessing.Pool() as pool:
            pool.starmap(self.plot_single_mha_spectrum, args)
        
        end = time.time()
        print("Elapsed Time: ", end-st)

    def get_QK(self, n_heads):
        """Obtains every Q,K pair in the network"""
        QK_list = []
        Q = None
        K = None
        unformatted_layer_name = None

        # self.weights is a collection of tuples acting as a tree structure
        for name, param in self.layer_list:
            if ("key" in name or "k_proj" in name) and "weight" in name:
                K = param.T
                K = K.reshape(K.shape[0], n_heads, -1).detach().cpu().numpy()
                unformatted_layer_name = name.split(".")
            elif ("query" in name or "q_proj" in name) and "weight" in name:
                Q = param.T
                Q = Q.reshape(Q.shape[0], n_heads, -1).detach().cpu().numpy()
            # Store if we found a key,query pair 
            if K is not None and Q is not None:
                name_list = []
                
                # Store tree up to the 'key' branch
                for term in unformatted_layer_name:
                    if term == "key" or term == "k_proj":
                        break
                    name_list.append(term)
                    
                formatted_layer_name = ("/").join(name_list)
                # print("Q,K found for layer: ", formatted_layer_name)
                QK_list.append((np.array(Q), np.array(K), formatted_layer_name))

                # Reset key, query pair for new search
                Q = None
                K = None
        return QK_list
    

    def get_V(self, n_heads):
        """Obtains every Q,K pair in the network"""
        V_list = []
        V = None
        unformatted_layer_name = None

        # self.weights is a collection of tuples acting as a tree structure
        for name, param in self.layer_list:
            if ("value" in name or "v_proj" in name) and "weight" in name:
                V = param.T
                V = V.reshape(V.shape[0], n_heads, -1).detach().cpu().numpy()
                unformatted_layer_name = name.split(".")
           
            if V is not None:
                name_list = []
                
                # Store tree up to the 'key' branch
                for term in unformatted_layer_name:
                    if term == "value" or term == "v_proj":
                        break
                    name_list.append(term)
                    
                formatted_layer_name = ("/").join(name_list)
                # print("Q,K found for layer: ", formatted_layer_name)
                V_list.append((np.array(V), formatted_layer_name))

                # Reset key, query pair for new search
                V = None
        return V_list # V is of shape (in_features, n_head, out_features). V should be left applied to a input. x @ V[:,h,:].