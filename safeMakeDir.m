function d = safeMakeDir(d)
      if ~isfolder(d)
           mkdir(d);
      end
end