"""from tqdm import tqdm
from pathlib import Path
import torchio as tio
import time
import numpy as np
path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/IsotropicDataset/"
inpPath = Path(path)

since = time.time()


count = 0
subjects = []
for file_name in tqdm(sorted(inpPath.glob("*T1*.nii.gz"))):
    subject = tio.Subject(image=tio.ScalarImage(file_name), label=0)
    moco = tio.transforms.Ghosting(num_ghosts=10, intensity=0.4, axis=0, restore=0)
    s = moco.apply_transform(subject)
    s["label"] = [int(4)]
    subjects.append(s)
    count += 1
print(count)
time_elapsed = time.time() - since
print('Single complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

import threading
import time

exitFlag = 0
subjects = []
class myThread (threading.Thread):
   def __init__(self, threadID, file_batch,  name):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.file_batch = file_batch
   def run(self):
      print ("\nStarting " + self.name)
      count = 0
      for file_name in sorted(self.file_batch):
          subject = tio.Subject(image=tio.ScalarImage(file_name), label=0)
          moco = tio.transforms.Ghosting(num_ghosts=10, intensity=0.4, axis=0, restore=0)
          s = moco.apply_transform(subject)
          s["label"] = [int(4)]
          subjects.append(s)
          print("\nThread : ", self.threadID , "  Count : ",count)
          count += 1
      print ("Exiting " + self.name)
      #return subjects
class test():
    def abc(self):
        since = time.time()
        file_index = []
        for file_name in tqdm(sorted(inpPath.glob("*T1*.nii.gz"))):
            file_index.append(file_name)

        no_threads = 10
        chunks = np.array_split(file_index, no_threads)
        i = 0
        thread = []
        for file_batch in chunks:
            i += 1
            thread.append(myThread(i, file_batch, "Thread"+str(i)))

        for thread_no in thread:
            thread_no.start()

        for thread_no in thread:
            thread_no.join()

        time_elapsed = time.time() - since
        print('Thread complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

A = test()
test.abc("")"""

for i in range(0,6):
    print(i)