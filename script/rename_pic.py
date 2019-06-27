# Pythono3 code to rename multiple 
# files in a directory or folder 

# importing os module 
import os 

# Function to rename multiple files 
def main(): 

	root = "/home/heisai/disk/HeisAI_data/face_pro/"

	for path, dirs, files in os.walk(root, followlinks=True):
		dirs.sort()
		files.sort()
		i = 0
		for fname in files:

			suffix = os.path.splitext(fname)[1].lower()
			if suffix == '.xml':
				i-=1
			strr = path.split("/")

			s = "%04d" % i
			dst = strr[-1] + "_" + s + suffix
			src = path+"/" + fname
			dst = path+"/" + dst

			# rename() function will
			# rename all the files
			os.rename(src, dst)
			i += 1


# Driver Code 
if __name__ == '__main__': 
	
	# Calling main() function 
	main() 

