echo "TransE YAGO4-19K"
python main.py -pipeline both -dataset YAGO4-18K -model TransE -ne 400 -save_each 10 -lr 0.001 -metrics all -reg 0.00001 -dim 100 -setting CWA -sem both
echo "TransH YAGO4-19K"
python main.py -pipeline both -dataset YAGO4-18K -model TransH -ne 400 -save_each 10 -lr 0.001 -metrics all -reg 0.00001 -dim 100 -setting CWA -sem both
echo "Distmult YAGO4-19K"
python main.py -pipeline both -dataset YAGO4-18K -model DistMult -batch_size 1024 -ne 400 -save_each 10 -lr 0.005 -metrics all -dim 100 -setting CWA -sem both
echo "ComplEx YAGO4-19K"
python main.py -pipeline both -dataset YAGO4-18K -model ComplEx -batch_size 1024 -ne 400 -save_each 10 -lr 0.001 -reg 0.01 -metrics all -dim 100 -setting CWA -sem both
echo "SimplE YAGO4-19K"
python main.py -pipeline both -dataset YAGO4-18K -model SimplE2 -ne 400 -batch_size 1024 -save_each 10 -lr 0.001 -reg 0.01 -metrics all -dim 100 -setting CWA -sem both
echo "ConvE YAGO4-19K"
python main.py -pipeline both -dataset YAGO4-18K -model ConvE -ne 400 -save_each 10 -lr 0.001 -reg 0.00001 -metrics all -setting CWA -hidden_size 9728 -embedding_shape1 20 -dim 200
echo "ConvKB YAGO4-19K"
python main.py -pipeline both -dataset YAGO4-18K -model ConvKB2D -ne 400 -save_each 10 -lr 0.001 -reg 0.00001 -metrics all -setting CWA -hid_convkb 100 -num_of_filters_convkb 128

echo "TransE YAGO3-37K"
python main.py -pipeline both -dataset YAGO3-37K -model TransE -ne 400 -save_each 10 -batch_size 256 -lr 0.0005 -metrics all -reg 0.00001 -dim 150 -setting CWA -sem both
echo "TransH YAGO3-37K"
python main.py -pipeline both -dataset YAGO3-37K -model TransH -ne 400 -save_each 10 -lr 0.0005 -batch_size 256 -metrics all -reg 0.00001 -dim 150 -setting CWA -sem both
echo "DistMult YAGO3-37K"
python main.py -pipeline both -dataset YAGO3-37K -model DistMult -batch_size 1024 -ne 400 -save_each 10 -lr 0.0005 -reg 0.00001 -metrics all -dim 150 -setting CWA -sem both
echo "ComplEx YAGO3-37K"
python main.py -pipeline both -dataset YAGO3-37K -model ComplEx -batch_size 1024 -ne 400 -save_each 10 -lr 0.001 -reg 0.01 -metrics all -dim 150 -setting CWA -sem both
echo "SimplE YAGO3-37K"
python main.py -pipeline both -dataset YAGO3-37K -model SimplE2 -ne 400 -batch_size 1024 -save_each 10 -lr 0.001 -reg 0.01 -metrics all -dim 150 -setting CWA -sem both
echo "ConvE YAGO3-37K"
python main.py -pipeline both -dataset YAGO3-37K -model ConvE -ne 400 -save_each 10 -lr 0.0005 -reg 0.00001 -metrics all -batch_size 256 -setting CWA -sem both -hidden_size 9728 -embedding_shape1 20 -dim 200
echo "ConvKB YAGO3-37K"
python main.py -pipeline both -dataset YAGO3-37K -model ConvKB2D -ne 400 -save_each 10 -lr 0.001 -reg 0.00001 -metrics all -setting CWA -batch_size 256 -sem both -hid_convkb 100 -num_of_filters_convkb 128

echo "TransE FB15K237"
python main.py -pipeline both -dataset FB15K237 -model TransE -ne 400 -save_each 10 -lr 0.001 -metrics all -reg 0.00001 -dim 100 -setting CWA -sem both
echo "TransH FB15K237"
python main.py -pipeline both -dataset FB15K237 -model TransH -ne 400 -save_each 10 -lr 0.001 -metrics all -reg 0.00001 -dim 100 -setting CWA -sem both
echo "Distmult FB15K237"
python main.py -pipeline both -dataset FB15K237 -model DistMult -batch_size 1024 -ne 400 -save_each 10 -lr 0.001 -metrics all -dim 100 -setting CWA -sem both
echo "ComplEx FB15K237"
python main.py -pipeline both -dataset FB15K237 -model ComplEx -batch_size 1024 -ne 400 -save_each 10 -lr 0.001 -reg 0.01 -metrics all -dim 100 -setting CWA -sem both
echo "SimplE FB15K237"
python main.py -pipeline both -dataset FB15K237 -model SimplE2 -ne 400 -batch_size 1024 -save_each 10 -lr 0.001 -reg 0.01 -metrics all -dim 100 -setting CWA -sem both
echo "ConvE FB15K237"
python main.py -pipeline both -dataset FB15K237 -model ConvE -ne 400 -save_each 10 -lr 0.001 -reg 0.00001 -metrics all -setting CWA -hidden_size 9728 -embedding_shape1 20 -dim 200 -sem both
echo "ConvKB FB15K237"
python main.py -pipeline both -dataset FB15K237 -model ConvKB2D -ne 400 -save_each 10 -lr 0.001 -reg 0.00001 -metrics all -setting CWA -hid_convkb 100 -num_of_filters_convkb 128 -sem both

echo "TransE DB93K"
python main.py -pipeline both -dataset DB93K -model TransE -ne 400 -save_each 10 -batch_size 256 -lr 0.0005 -metrics all -reg 0.00001 -dim 200 -setting CWA -sem both
echo "TransH DB93K"
python main.py -pipeline both -dataset DB93K -model TransH -ne 400 -save_each 10 -lr 0.0005 -batch_size 256 -metrics all -reg 0.00001 -dim 200 -setting CWA -sem both
echo "DistMult DB93K"
python main.py -pipeline both -dataset DB93K -model DistMult -batch_size 1024 -ne 400 -save_each 10 -lr 0.0005 -reg 0.00001 -metrics all -dim 200 -setting CWA -sem both
echo "ComplEx DB93K"
python main.py -pipeline both -dataset DB93K -model ComplEx -batch_size 1024 -ne 400 -save_each 10 -lr 0.001 -reg 0.01 -metrics all -dim 200 -setting CWA -sem both
echo "SimplE2 DB93K"
python main.py -pipeline both -dataset DB93K -model SimplE2 -ne 400 -batch_size 1024 -save_each 10 -lr 0.001 -reg 0.01 -metrics all -dim 200 -setting CWA -sem both
echo "ConvE DB93K"
python main.py -pipeline both -dataset DB93K -model ConvE -ne 400 -save_each 10 -lr 0.001 -reg 0.00001 -metrics all -setting both -sem both -hidden_size 9728 -embedding_shape1 20 -dim 200
echo "ConvKB DB93K"
python main.py -pipeline both -dataset DB93K -model ConvKB2D -ne 400 -save_each 10 -lr 0.001 -reg 0.00001 -metrics all -setting both -sem both -hid_convkb 100 -num_of_filters_convkb 128

echo "TransE Codex-S"
python main.py -pipeline both -dataset Codex-S -model TransE -batch_size 128 -ne 400 -save_each 10 -lr 0.001 -metrics all -reg 0.00001 -dim 100 -setting CWA -sem extensional
echo "TransH Codex-S"
python main.py -pipeline both -dataset Codex-S -model TransH -batch_size 128 -ne 400 -save_each 10 -lr 0.001 -metrics all -reg 0.00001 -dim 100 -setting CWA -sem extensional
echo "Distmult Codex-S"
python main.py -pipeline both -dataset Codex-S -model DistMult -batch_size 1024 -ne 400 -save_each 10 -lr 0.001 -metrics all -dim 100 -setting CWA -sem extensional
echo "ComplEx Codex-S"
python main.py -pipeline both -dataset Codex-S -model ComplEx -batch_size 1024 -ne 400 -save_each 10 -lr 0.001 -metrics all -dim 100 -setting CWA -sem extensional
echo "SimplE Codex-S"
python main.py -pipeline both -dataset Codex-S -model SimplE2 -ne 400 -batch_size 1024 -save_each 10 -lr 0.001 -metrics all -dim 100 -setting CWA -sem extensional
echo "ConvE"
python main.py -pipeline both -dataset Codex-S -model ConvE -batch_size 512 -ne 400 -save_each 10 -lr 0.001 -reg 0.00001 -metrics all -setting CWA -hidden_size 9728 -embedding_shape1 20 -sem extensional
echo "ConvKB"
python main.py -pipeline both -dataset Codex-S -model ConvKB2D -batch_size 512 -ne 400 -save_each 10 -lr 0.001 -reg 0.00001 -metrics all -setting CWA -hid_convkb 100 -num_of_filters_convkb 128 -sem extensional

echo "TransE Codex-M"
python main.py -pipeline both -dataset Codex-M -model TransE -batch_size 128 -ne 400 -save_each 10 -lr 0.001 -metrics all -reg 0.00001 -dim 100 -setting CWA -sem extensional
echo "TransH Codex-M"
python main.py -pipeline both -dataset Codex-M -model TransH -batch_size 128 -ne 400 -save_each 10 -lr 0.001 -metrics all -reg 0.00001 -dim 100 -setting CWA -sem extensional
echo "Distmult Codex-M"
python main.py -pipeline both -dataset Codex-M -model DistMult -batch_size 1024 -ne 400 -save_each 10 -lr 0.001 -metrics all -dim 100 -setting CWA -sem extensional
echo "ComplEx Codex-M"
python main.py -pipeline both -dataset Codex-M -model ComplEx -batch_size 1024 -ne 400 -save_each 10 -lr 0.001 -metrics all -dim 100 -setting CWA -sem extensional
echo "SimplE Codex-M"
python main.py -pipeline both -dataset Codex-M -model SimplE2 -ne 400 -batch_size 1024 -save_each 10 -lr 0.001 -metrics all -dim 100 -setting CWA -sem extensional
echo "ConvE Codex-M"
python main.py -pipeline both -dataset Codex-M -model ConvE -batch_size 512 -ne 400 -save_each 10 -lr 0.001 -reg 0.00001 -metrics all -setting CWA -hidden_size 9728 -embedding_shape1 20 -sem extensional
echo "ConvKB Codex-M"
python main.py -pipeline both -dataset Codex-M -model ConvKB2D -batch_size 512 -ne 400 -save_each 10 -lr 0.001 -reg 0.00001 -metrics all -setting CWA -hid_convkb 100 -num_of_filters_convkb 128 -sem extensional

echo "TransE WN18RR"
python main.py -pipeline both -dataset WN18RR -model TransE -ne 400 -save_each 10 -lr 0.001 -metrics all -reg 0.00001 -dim 100 -setting CWA -sem extensional
echo "TransH WN18RR"
python main.py -pipeline both -dataset WN18RR -model TransH -ne 400 -save_each 10 -lr 0.001 -metrics all -reg 0.00001 -dim 100 -setting CWA -sem extensional
echo "Distmult WN18RR"
python main.py -pipeline both -dataset WN18RR -model DistMult -batch_size 1024 -ne 400 -save_each 10 -lr 0.001 -metrics all -dim 100 -setting CWA -sem extensional
echo "ComplEx WN18RR"
python main.py -pipeline both -dataset WN18RR -model ComplEx -batch_size 1024 -ne 400 -save_each 10 -lr 0.001 -metrics all -dim 100 -setting CWA -sem extensional
echo "SimplE WN18RR"
python main.py -pipeline both -dataset WN18RR -model SimplE2 -ne 400 -batch_size 1024 -save_each 10 -lr 0.001 -metrics all -dim 100 -setting CWA -sem extensional
echo "ConvE WN18RR"
python main.py -pipeline both -dataset WN18RR -model ConvE -ne 400 -save_each 10 -lr 0.001 -reg 0.00001 -metrics all -setting CWA -hidden_size 9728 -embedding_shape1 20 -sem extensional
echo "ConvKB WN18RR"
python main.py -pipeline both -dataset WN18RR -model ConvKB2D -ne 400 -save_each 10 -lr 0.001 -reg 0.00001 -metrics all -setting CWA -hid_convkb 100 -num_of_filters_convkb 128 -sem extensional