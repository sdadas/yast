package pl.sdadas.fstlexicon.command;

import pl.sdadas.fstlexicon.fst.FSTSearch;

import java.util.Map;

/**
 * @author Sławomir Dadas
 */
public interface Command {

    boolean readable(String input);

    void read(String input, Map<String, FSTSearch<Long>> dictionaries);
}
