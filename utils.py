import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2


def savefig(img,outmask,truemask,path,reconimg):
    grid = ImageGrid(
        fig=plt.figure(figsize=(16, 4)),
        rect=111,
        nrows_ncols=(1, 4),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.15,
    )
    img=img.permute(0,2,3,1)
    img = img[0, :, :, :]
    img = img.cpu().numpy()
    img = img* 220
    img = img.astype('uint8')
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    reconimg=reconimg.permute(0,2,3,1)
    reconimg = reconimg[0, :, :, :]
    reconimg = reconimg.cpu().numpy()
    reconimg = reconimg* 220
    reconimg = reconimg.astype('uint8')
    reconimg=cv2.cvtColor(reconimg,cv2.COLOR_BGR2RGB)

    grid[0].imshow(img)
    grid[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    grid[0].set_title("Input Image", fontsize=14)

    grid[1].imshow(reconimg)
    grid[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    grid[1].set_title("Recon Image", fontsize=14)

    grid[2].imshow(truemask)
    grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    grid[2].set_title("GroundTruth", fontsize=14)

    grid[3].imshow(img)
    im = grid[3].imshow(outmask, alpha=0.3, cmap="jet")
    grid[3].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    grid[3].cax.colorbar(im)
    grid[3].cax.toggle_label(True)
    grid[3].set_title("Anomaly Map", fontsize=14)

    plt.savefig(path, bbox_inches="tight")
    plt.close()


