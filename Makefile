all: datasets

datasets: 
	wget https://zenodo.org/record/5084812/files/PSB2.zip?download=1
	unzip PSB2.zip
	rm PSB2.zip