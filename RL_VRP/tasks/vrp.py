"""Defines the main task for the VRP.

The VRP is defined by the following traits:
    1. Each city has a demand in [1, 9], which must be serviced by the vehicle
    2. Each vehicle has a capacity (depends on problem), the must visit all cities
    3. When the vehicle load is 0, it __must__ return to the depot to refill
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print(torch.cuda.is_available()) # true 查看GPU是否可用
print(torch.cuda.device_count()) #GPU数量， 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VehicleRoutingDataset(Dataset):
    def __init__(self, num_samples, input_size, max_load=40, max_demand=9,
                 seed=None):
        super(VehicleRoutingDataset, self).__init__()
        
        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self.max_load = max_load
        self.max_demand = max_demand
        
        # Depot location will be the first node in each
        ### the number of transfer point
        num_transfer = int(input_size/10)
        ### the number of customer + transfer point + depot
        input_shape =input_size+ num_transfer + 1
        locations = torch.rand((num_samples, 2, input_shape))
        self.static = locations
        self.transfer_num = num_transfer
        ###Record the coespondence of transfer point for every customer 
        ###depot, transfer point = 0
        ###customer = index of transfer point
        ###I define the index of depot is the 0 and 
        ###the index of transfer point are following behind.
        correspond = torch.zeros([num_samples,2,input_shape])
        correspond[:,1,:] = 1
        ###construct a tensor to record the point which used crowdsource
        self.render_record = torch.zeros([num_samples,input_shape])
        
        for i in range(1,num_transfer+1):
            ###calculate distance
            transfer_point = locations[:,:,i].unsqueeze(2)
            distance_2 = (locations-transfer_point).pow(2).sum(axis = 1)
            
            ###decide which customers are in the scope
            if distance_2.lt(0.04).any() and distance_2.lt(correspond[:,1]).any():
                Inscope = torch.logical_and(distance_2.lt(0.04), distance_2.lt(correspond[:,1]))
                correspond[Inscope,0] = i
                correspond[Inscope,1] = distance_2[Inscope]
        correspond[:,0,0:(num_transfer+1)] = 0
        correspond[:,1,0:(num_transfer+1)] = 1
        self.transfer_correspond = correspond
        
        # All states will broadcast the drivers current load
        # Note that we only use a load between [0, 1] to prevent large
        # numbers entering the neural network
        dynamic_shape = (num_samples, 1, input_shape)           ### adjust the number of nodes
        loads = torch.full(dynamic_shape, 1.)
        
        # All states will have their own intrinsic demand in [1, max_demand), 
        # then scaled by the maximum load. E.g. if load=10 and max_demand=30, 
        # demands will be scaled to the range (0, 3)
        demands = torch.randint(1, max_demand + 1, dynamic_shape)
        demands = demands / float(max_load)
        demands = np.round(demands,decimals=4)
        
        
        demands[:, 0, 0:(num_transfer+1)] = 0  # depot starts with a demand of 0 
                                               ### initial the demands of transfer point
                                               
        for transfer in range(1,num_transfer+1):
            ###record the demands of transfer point
            demand_fake = torch.zeros([num_samples,1,input_shape])  
            transfer_scope = correspond[:,0,:].eq(transfer)            
            demand_fake[transfer_scope,0] = demands.squeeze(1)[transfer_scope]
            demands[:,0,transfer] = demand_fake.sum(2).squeeze(1)
        
        
        self.dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))
        
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1],
                self.transfer_correspond[idx], self.render_record[idx])

    def update_mask(self, mask, dynamic, chosen_idx=None):
        """Updates the mask used to hide non-valid states.

        Parameters
        ----------
        dynamic: torch.autograd.Variable of size (1, num_feats, seq_len)
        """

        # Convert floating point to integers for calculations
        loads = dynamic.data[:, 0]  # (batch_size, seq_len)
        demands = dynamic.data[:, 1]  # (batch_size, seq_len)
        
        # If there is no positive demand left, we can end the tour.
        # Note that the first node is the depot, which always has a negative demand
        if demands.eq(0).all():
            return demands * 0.

        # Otherwise, we can choose to go anywhere where demand is > 0
        new_mask = demands.ne(0) * demands.lt(loads) 
        # We should avoid traveling to the depot back-to-back 
        # if not the same(no go home), return true(1).
        repeat_home = chosen_idx.ne(0)
        #same return false, diff return true
        
        if repeat_home.any():
            new_mask[repeat_home.nonzero(), 0] = 1.
        if (~repeat_home).any():
            new_mask[(~repeat_home).nonzero(), 0] = 0.

        # ... unless we're waiting for all other samples in a minibatch to finish
        has_no_load = loads[:, 0].eq(0).float()
        has_no_demand = demands[:, 1:].sum(1).eq(0).float()
        
        combined = (has_no_load + has_no_demand).gt(0)
        
        if combined.any():
            new_mask[combined.nonzero(), 0] = 1.
            new_mask[combined.nonzero(), 1:] = 0.
        num = self.transfer_num+1
        new_mask[:,1:num] = new_mask[:,1:num] * demands[:,1:num].gt(0.00001)
        
        return new_mask.float()

    def update_dynamic(self, dynamic, chosen_idx, transfer_correspond,render_record):             ###transfer_correspond, render_record
        """Updates the (load, demand) dataset values."""

        # Update the dynamic elements differently for if we visit depot vs. a city
        #same return false, diff return true
        
        visit = chosen_idx.ne(0)
        depot = chosen_idx.eq(0)
        
        # Clone the dynamic variable so we don't mess up graph
        all_loads = dynamic[:, 0].clone()
        all_demands = dynamic[:, 1].clone()
        
        
        load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
        demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))
        
        # Across the minibatch - if we've chosen to visit a city, try to satisfy
        # as much demand as possible
        if visit.any():
            
            
            new_load = torch.clamp(load - demand, min=0)
            new_demand = torch.clamp(demand - load, min=0)
            # Broadcast the load to all nodes, but update demand seperately
            visit_idx = visit.nonzero().squeeze()
            all_loads[visit_idx] = new_load[visit_idx]
            all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(-1)
            all_demands[visit_idx, 0] = -1. + new_load[visit_idx].view(-1)
            
            ###calculate the quantity fo demand which the trucks transport
            ###storage the self value
            diff_demand = demand - new_demand
            transfer = self.transfer_num
            transfer_correspond = transfer_correspond[:,0,:]
            
            ###calculate the demand about the transfer points
            
            for i in range(1,transfer+1):
                
                ###pocesss the demands of transfer which the correspond customer is went through 
                cus_transfer = transfer_correspond[visit,chosen_idx[visit]].eq(i)
                if visit_idx.shape == torch.Size([]):
                    visit_idx = visit_idx.unsqueeze(0) 
                cus_transfer_idx = visit_idx[cus_transfer]
                all_demands[cus_transfer_idx , i] = all_demands[cus_transfer_idx , i] - diff_demand[cus_transfer_idx].view(-1)
               
                
               ###pocess the demands of transfer point when the transfer point is directly went through by truck
                visit_transfer = chosen_idx[visit].eq(i)
                visit_transfer_idx = visit_idx[visit_transfer]
                zero_transfer_idx =  visit_transfer_idx[all_demands[visit_transfer_idx,i].eq(0)] 
                cus_correspond = transfer_correspond[zero_transfer_idx].eq(i)
                flag_demands = all_demands[zero_transfer_idx]
                flag_demands[cus_correspond] = 0.
                all_demands[zero_transfer_idx] = flag_demands
                render_record[zero_transfer_idx] = transfer_correspond[zero_transfer_idx]
                
                ### the situation that the load cant satisfy the demand of trsferpoint
                notzero_transfer_idx =  visit_transfer_idx[all_demands[visit_transfer_idx,i].ne(0)]
                cus_correspond2 = transfer_correspond[notzero_transfer_idx].eq(i)
                flag = 0
                for j in cus_correspond2:
                    
                    j = j.nonzero().squeeze()
                    loading = float(diff_demand[int(notzero_transfer_idx[flag])])
                    for k in j :
                        if loading - all_demands[notzero_transfer_idx[flag],int(k)] >= 0:
                            loading = loading - all_demands[notzero_transfer_idx[flag],int(k)]
                            all_demands[notzero_transfer_idx[flag],int(k)] = 0
                            render_record[notzero_transfer_idx[flag],int(k)] = i
                        else:
                            all_demands[notzero_transfer_idx[flag],int(k)] = all_demands[notzero_transfer_idx[flag],int(k)] - loading
                            render_record[notzero_transfer_idx[flag],int(k)] = i
                            flag = flag +1
                            break
        # Return to depot to fill vehicle load
        if depot.any():
            all_loads[depot.nonzero().squeeze()] = 1.
            all_demands[depot.nonzero().squeeze(), 0] = 0.
        
        
        tensor = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
        
        return torch.tensor(tensor.data, device=dynamic.device)


def reward(num_nodes,transfer_correspond,static, tour_indices):
    """
    Euclidean distance between all cities / nodes given by tour_indices
    """
    
    
    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)
    # Ensure we're always returning to the depot - note the extra concat
    # won't add any extra loss, as the euclidean distance between consecutive
    # points is 0
    start = static.data[:, :, 0].unsqueeze(1)
    y = torch.cat((start, tour, start), dim=1)
    # Euclidean distance between each consecutive point
    
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    
    tour_len = tour_len.sum(1).unsqueeze(1)
    
    
    ###judge which sample pass the transfer point and plus the distance between customer and transfer point
    for i in range(1,int(num_nodes/10)+1):
     
     transfer_dis = transfer_correspond[:,1,:].to(device)
     transfer_dis_idx = transfer_correspond[:,0,:].ne(i)
     transfer_dis[transfer_dis_idx] = 0.
     transfer_dis = transfer_dis.sum(1).unsqueeze(1)
     went_transfer = tour_indices.eq(i).any(1).unsqueeze(1)
     went_transfer_idx = went_transfer.nonzero()[:,0].squeeze()
     tour_len[went_transfer_idx] = tour_len[went_transfer_idx] + 0.7*transfer_dis[went_transfer_idx] 
    tour_len = tour_len.squeeze(1)

    
    return tour_len

def render(render_record, static, tour_indices, save_path):
    """Plots the found solution."""
    
    
    plt.close('all')
    
    num_plots = 2
    
    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots)
    
    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]
    
    for i, ax in enumerate(axes):

        # Convert the indices back into a tour

        idx = tour_indices[i]
        
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)
        render_correspond = render_record[i]
        
        
        if i == 1 :
            #print('draw : ',render_correspond)
            print('route : ',tour_indices[1,0:25])
        
        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()
        
        start = static[i, :, 0].cpu().data.numpy()
        
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))
       
        # Assign each subtour a different colour & label in order traveled
       
        idx = np.hstack((0, tour_indices[i].cpu().numpy().flatten(), 0))
        where = np.where(idx == 0)[0]
        
        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            if low + 1 == high:
                continue

            ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=j)
        
        node_num = int(static.shape[2])
        print(i,' : ')
        for k in range(1,int((node_num-1)/11 + 1)):
            transfer = static[i, :, k].cpu().data.numpy()
            ax.scatter(transfer[0], transfer[1], s=20, c='k', marker='P', zorder=3)
            if tour_indices[i].ne(k).prod():
                continue
            else:
                print(k,"sss:")
            for iq ,cor in enumerate(render_correspond):
                if cor == k:
                    if tour_indices[i].ne(iq).prod():
                        print(iq)
                        q = static[i, :, iq].cpu().data.numpy()
                        ax.scatter(q[0], q[1], s=4, c='r', zorder=2)
                        ax.plot([transfer[0],q[0]], [transfer[1],q[1]], zorder=1, label=k+5,linestyle='dashed',dashes=(0.6,0.4))
                    else:
                        continue
            
        ax.legend(loc="upper right", fontsize=3, framealpha=0.5)
        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    print(save_path)
    plt.savefig(save_path, bbox_inches='tight', dpi=1200)

'''
def render(render_record,static, tour_indices, save_path):
    """Plots the found solution."""
    
    
    plt.close('all')
    
    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')
    
    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]
    
    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        
        idx = tour_indices[i]
        
        print('route : ',idx)
        
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)
        render_correspond = render_record[i]
        
        print('draw : ',render_correspond)
        
        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()
        
        start = static[i, :, 0].cpu().data.numpy()
        
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))
       
        # Assign each subtour a different colour & label in order traveled
       
        idx = np.hstack((0, tour_indices[i].cpu().numpy().flatten(), 0))
        where = np.where(idx == 0)[0]
        
        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            if low + 1 == high:
                continue

            ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=j)
        node_num = int(static.shape[2])
        for k in range(1,int((node_num-1)/11 + 1)):
            transfer = static[i, :, k].cpu().data.numpy()
            ax.scatter(transfer[0], transfer[1], s=20, c='k', marker='P', zorder=3)
            for iq ,cor in enumerate(render_correspond):
                if cor == k:
                    q = static[i, :, iq].cpu().data.numpy()
                    ax.scatter(q[0], q[1], s=4, c='r', zorder=2)
                    ax.plot([transfer[0],q[0]], [transfer[1],q[1]], zorder=1, label=k+5,linestyle='dashed')    
                   
            
        ax.legend(loc="upper right", fontsize=3, framealpha=0.5)
        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        print(save_path)
        plt.savefig(save_path, bbox_inches='tight', dpi=200)


    plt.tight_layout()
    print(save_path)
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
'''

'''
def render(static, tour_indices, save_path):
    """Plots the found solution."""

    path = 'C:/Users/Matt/Documents/ffmpeg-3.4.2-win64-static/bin/ffmpeg.exe'
    plt.rcParams['animation.ffmpeg_path'] = path

    plt.close('all')

    num_plots = min(int(np.sqrt(len(tour_indices))), 3)
    fig, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                             sharex='col', sharey='row')
    axes = [a for ax in axes for a in ax]

    all_lines = []
    all_tours = []
    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        cur_tour = np.vstack((x, y))

        all_tours.append(cur_tour)
        all_lines.append(ax.plot([], [])[0])

        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

    from matplotlib.animation import FuncAnimation

    tours = all_tours

    def update(idx):

        for i, line in enumerate(all_lines):

            if idx >= tours[i].shape[1]:
                continue

            data = tours[i][:, idx]

            xy_data = line.get_xydata()
            xy_data = np.vstack((xy_data, np.atleast_2d(data)))

            line.set_data(xy_data[:, 0], xy_data[:, 1])
            line.set_linewidth(0.75)

        return all_lines

    anim = FuncAnimation(fig, update, init_func=None,
                         frames=100, interval=200, blit=False,
                         repeat=False)

    anim.save('line.mp4', dpi=160)
    plt.show()

    import sys
    sys.exit(1)
'''
