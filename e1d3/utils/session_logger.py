import os
import sys


def show_progress(batch_number, total_batches, metrics_string=''):
    """
        Displays progress of a training/testing loop.
        Args:
            batch_number: 'int', current iteration.
            total_batches: 'int', total iterations.
            metrics_string: 'str', anything else to print (usually a metric).
        Reference: https://stackoverflow.com/a/15860757/6597334
        """
    bar_length = 40  # modify to change the length of the progress bar
    status = ""
    if batch_number > total_batches:
        raise Exception('Batch Number exceeds Total Batches')
    elif batch_number == total_batches:
        status = "Done...\r\n"
    progress = batch_number * 1.0 / total_batches
    block = int(round(bar_length * progress))
    text = "\rBatch:{2}/{3}: [{0}] {1}%  {4}  {5}".format(
        "#" * block + "-" * (bar_length - block), '%3d' % (progress * 100),
        batch_number, total_batches, metrics_string, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def log_configuration(save_location, config_file):
    if not os.path.isdir(save_location):
        os.mkdir(save_location)

    with open(config_file, 'r') as txtfile_read:
        data = txtfile_read.read()
        # print(data)     # print configurations on screen
        with open(os.path.join(save_location, os.path.split(config_file)[-1]),
                  'w') as txtfile_write:
            txtfile_write.write(data)  # write configurations to .txt file
