package pl.sdadas.fstlexicon.command;

import org.apache.commons.lang3.StringUtils;
import pl.sdadas.fstlexicon.fst.FSTSearch;

import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author Sławomir Dadas
 */
public class ListDictionaries implements Command {

    @Override
    public boolean readable(String input) {
        return StringUtils.strip(input).equalsIgnoreCase("--list");
    }

    @Override
    public void read(String input, Map<String, FSTSearch<Long>> dictionaries) {
        String output = dictionaries.keySet().stream().sorted().collect(Collectors.joining(" "));
        System.err.println(output);
    }
}
