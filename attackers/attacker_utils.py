import numpy as np


def get_label_dict(name_idx_dict):
    return {value:key for key,value in name_idx_dict.items()}


def save_tensor2txt(fn,tensor,duty_label,org_label,global_idx):
    # tensor size: NxD
    # dict: {label_name: label_idx}

    data = tensor.numpy()

    data_fn = "{}/{}_{}_{:04d}.txt".format(fn,
                                       duty_label,org_label,
                                   global_idx)
    np.savetxt(data_fn,data,delimiter=',',fmt='%.6f')


def save_tensor2obj(fn,tensor,duty_label,org_label,global_idx):
    # tensor size: NxD
    data = tensor.numpy()

    use_normals = (data.shape[1] == 6)

    data_fn = "{}/{}_{}_{:04d}.obj".format(fn,
                                       duty_label,org_label,
                                   global_idx)

    fout = open(data_fn,'w')

    for i in range(data.shape[0]):
        if not use_normals:
            fout.write('v %f %f %f\n' % (data[i][0], data[i][1],data[i][2]))
        else :
            fout.write('v %f %f %f %f %f %f\n' % (data[i][0],
                                                  data[i][1],data[i][2],data[i][3],data[i][4],data[i][5]))

    fout.close()
