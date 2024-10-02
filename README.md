# CNN
A CNN using SGD with early stopping compares the difference between the local minimum within a ‘patience’ window and a global maximum. It then compares this difference with min_delta to determine if the changes exceed this minimum value. If not, the execution stops, and the best weights are restored if needed. Additionally, there is an option to set a minimum desired accuracy or loss.
