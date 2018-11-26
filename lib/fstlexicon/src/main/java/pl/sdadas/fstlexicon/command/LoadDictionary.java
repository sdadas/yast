package pl.sdadas.fstlexicon.command;

import org.apache.commons.lang3.StringUtils;
import org.apache.lucene.util.fst.FST;
import pl.sdadas.fstlexicon.fst.FSTSearch;
import pl.sdadas.fstlexicon.fst.FSTUtils;

import java.io.File;
import java.util.Map;

/**
 * @author Sławomir Dadas
 */
public class LoadDictionary implements Command {

    @Override
    public boolean readable(String input) {
        return StringUtils.startsWith(input.trim(), "--load");
    }

    @Override
    public void read(String input, Map<String, FSTSearch<Long>> dictionaries) {
        String line = input.trim();
        String[] args = StringUtils.split(line, " ", 3);
        if(args.length < 3) {
            String msg = "Cannot parse line, format for loading should be '--load dictionary_name dictionary_path'";
            throw new IllegalStateException(msg);
        }
        String name = StringUtils.lowerCase(args[1]);
        String path = args[2];
        File fstFile = new File(path);
        FST<Long> fst = FSTUtils.load(fstFile, Long.class);
        dictionaries.put(name, new FSTSearch<>(fst));
    }
}
