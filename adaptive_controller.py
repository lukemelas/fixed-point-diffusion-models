import torch

# calculate the norm (modified) of the delta
def batched_distance(dt):
    dist = torch.norm(dt, p = 2, dim = (1, 2))
    dist = torch.mean(dist)
    return dist

class LinearController():
    def __init__(self, budget, tot_steps, ratio = [0.5, 1.5], type = "increasing") -> None:
        if type == "increasing":
            self.ratio_list = torch.linspace(ratio[0], ratio[1], tot_steps)
        elif type == "decreasing":
            self.ratio_list = torch.linspace(ratio[1], ratio[0], tot_steps)
        elif type == "fixed":
            self.ratio_list = torch.ones(tot_steps)

        assert len(self.ratio_list) == tot_steps
        self.budget_list_float = 1 + (budget - tot_steps) * self.ratio_list / torch.sum(self.ratio_list)

        rounding_threshold = len(self.budget_list_float) // 2
        while True:
            self.budget_list = torch.zeros_like(self.budget_list_float)
            for i in range(len(self.budget_list_float)):
                # in the first half, round up
                # in the second half, round down
                # this ensures the sum of the budget is roughly equal to the total budget
                if i < rounding_threshold:
                    self.budget_list[i] = torch.ceil(self.budget_list_float[i])
                else:
                    self.budget_list[i] = torch.floor(self.budget_list_float[i])
            if torch.sum(self.budget_list) <= budget:
                break
            rounding_threshold -= 1
        print(f"fixed the rounding issue! in ", type)
        
        assert torch.sum(self.budget_list) <= budget

        self.pointer = len(self.budget_list) - 1
        self.threshold = None
        self.cost = None
        self.lowerbound = None
        self.upperbound = None
    
    def init_image(self):
        self.pointer = len(self.budget_list) - 1
        print(f"in Fix controller, budget list: {self.budget_list}")
    def end_image(self):
        pass
    def update(self):
        return True
    def get(self):
        ret = self.budget_list[self.pointer]
        self.pointer -= 1
        return int(ret)


# define a class to analyis threshold
class FixedController():
    def __init__(self, threshold) -> None:
        self.threshold = threshold
        self.cost = 0
    
    def get(self):
        return self.threshold

    def add_cost(self, cost):
        self.cost += cost
    
            

# class ThresholdController():
#     def __init__(self, budget, ratio_delta = 0) -> None:
#         self.budget = budget
#         self.threshold = 100.0
#         self.last_threshold = -1
#         self.lowerbound = 0.9 * budget
#         self.upperbound = 1.0 * budget
#         self.success_list = []
#         self.all_threshold_list = []
#         self.try_count = []
#         self.costs = []

#         self.delta_init_ratio = 0.03
#         self.delta = self.threshold * self.delta_init_ratio
#         self.count = 0
#         self.thresholds = []
#         self.cost = 0
#         self.pivot = False

#         self.max_count = 20 # the max number of threshold update on each batch.
        
#         # retrieve the number
#         self.upperbound_ratio = 1.0 + ratio_delta
#         self.lowerbound_ratio = 1.0 - ratio_delta

#     def init_image(self):
#         if len(self.all_threshold_list) != 0:
#             self.threshold = self.success_list[len(self.success_list) // 2]

#         self.last_threshold = -1
#         self.delta = self.threshold * self.delta_init_ratio
#         self.count = 0
#         self.cost = 0
#         self.pivot = False
#         self.thresholds = []
#         return self.threshold

#     def end_image(self):
#         self.success_list.append(self.threshold)
#         self.try_count.append(self.count)
#         self.all_threshold_list.append(self.thresholds)
#         self.success_list.sort()

#     def get(self):
#         return self.threshold

#     def add_cost(self, cost):
#         self.cost += cost

#     def update(self):
#         # after finishing an image
#         self.thresholds.append(self.threshold)
#         # print(f"in update: thres={self.threshold}, last thres={self.last_threshold}")
#         if self.cost > self.upperbound:
#             if self.last_threshold != -1 and self.threshold < self.last_threshold: # decrease, increase
#                 self.pivot = True
#         if self.cost < self.lowerbound:
#             if self.last_threshold != -1 and self.threshold > self.last_threshold: # increase, decrease
#                 self.pivot = True
        
#         if self.pivot:
#             self.delta /= 2
#         else:
#             self.delta *= 2


#         self.last_threshold = self.threshold
#         if self.cost > self.upperbound:
#             self.threshold = self.threshold + self.delta
#         if self.cost < self.lowerbound:
#             self.threshold = self.threshold - self.delta

#         passed = (self.lowerbound <= self.cost) and (self.cost <= self.upperbound)
#         # TODO: special case
#         if self.count > self.max_count and self.cost <= self.lowerbound:
#             passed = True
#         print(f"passed={passed}")
            
#         if passed:
#             self.costs.append(self.cost)
        
#         self.cost = 0
#         self.count += 1

#         return passed
    
#     def output(self):
#         """
#         success_list: list of threshold, (B)
#         all_threshold_list: list of list of threshold, (B, *)
#         try_count: list of try count, (B)
#         """
#         return self.success_list, self.all_threshold_list, self.try_count, self.costs

            
            

