### Finding optimum values for L1-penalty-weight & L2-weight-decay
    - The appropriate values for l1 & l2 related weights were found after coarsely sweeping
      thru values from 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001 first.
    - For l1-weight, with l1-alone enabled, the 0.00001 region was found with better values
      for the validation accuracy vis-a-vis training accuracies.
    - Much finer sweeping in the 0.00001 region, led to the value: 0.000025, which seems to
      be having best accuracy values.
    - Similarly, For l2-weight,  with l2-alone enabled, the 0.0001 region was found with better
      values for the validation accuracy vis-a-vis training accuracies.
    - Much finer sweeping in the 0.0001 region, led to the value: 0.0002125, which seems to
      be having best accuracy values.
    - For individual models, either or both of them will be enabled/disbaled using some
      related flags defined ahead in this notebook
    - Please note, L2 regaularization implementation is in-built with torch. 
