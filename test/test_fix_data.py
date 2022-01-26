
from tfrecord_loader import TFRecordNewInputs
import time

# for index_path in ["data/deleteme.index", "data/twitter9.train.index"]:
# for index_path in ["data/twitter9_LOCAL.train.index"]:
# for index_path in ["data/twitter16_LOCAL.train.index", "data/twitter16_LOCAL.val.index"]:
for index_path in ["data/twitter14_LOCAL.train.index", "data/twitter14_LOCAL.val.index"]:
    print(index_path)
    t0 = time.time()
    train_dataset = TFRecordNewInputs(index_path,
                                    batch_size=(
                                        5000,
                                        2),
                                    sample_size=4096,
                                    restore_state=None)
    # This will produce samples of (256,1,2049)
    t1 = time.time()
    print(f"init time: {t1-t0} sec")
    samples = train_dataset.get_samples()
    for i,sample in enumerate(train_dataset.sample_once()):
        # print(sample.shape)
        # if i == 800:
        #     break
        if sample.shape != (256, 2, 2049):
            print(f"sample [{i}] is {sample.shape}")
            print([(i,z) for (i,z) in enumerate([x.shape[0] for [x,y] in sample]) if z != 2049])
            print([(i,z) for (i,z) in enumerate([y.shape[0] for [x,y] in sample]) if z != 2049])

# Sample once:
# sample = next(train_dataset.sample_once())
# import numpy as np
# x = np.array(sample[0:16,0,:])
# print(x.shape)
