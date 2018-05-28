from PIL import Image


def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def find_between_r( s, first, last ):
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""


def make_square(im, min_size=256, fill_color=(0, 0, 0,0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, ((size - x) / 2, (size - y) / 2))
    new_im=new_im.convert('RGB')
    return new_im


basewidth = 256

with open('images.txt') as f:
   for line in f:
       line=find_between_r(line, "/", "\n")
       print(line)
       img = Image.open('/mnt/md0/jgarcia/pose-hg-train/data/mpii/images/'+line)
       wpercent = (basewidth/float(img.size[0]))
       hsize = int((float(img.size[1])*float(wpercent)))
#       img = img.resize((basewidth,hsize), Image.ANTIALIAS)
       img=img.resize(256,256)
#       img = make_square(img, 256, (0,0,0,0))
       img.save('/mnt/md0/jgarcia/pose-hg-train/data/mpii/images_256/'+line)


#with open('train_images.txt') as f:
#   for line in f:
#       name=find_between_r( line, "/0", ".jpg" )
#       file=open("keypoint_train_ascii.txt","a")
#       javi=[ord(c) for c in name]
#       print javi
       #file.write("%s", javi)
       #file.close()

