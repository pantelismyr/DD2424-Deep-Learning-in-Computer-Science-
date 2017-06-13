book_fname = 'Dataset/goblet_book.txt';
fid = fopen(book_fname, 'r');
book_data = fscanf(fid, '%c');
fclose(fid);

C = unique(book_data)

char_to_ind = containers.Map('KeyType', 'char', 'ValueType', 'int32')