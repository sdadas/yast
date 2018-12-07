package pl.sdadas.fstlexicon;

import org.apache.commons.lang3.StringEscapeUtils;
import pl.sdadas.fstlexicon.command.Command;
import pl.sdadas.fstlexicon.command.ListDictionaries;
import pl.sdadas.fstlexicon.command.LoadDictionary;
import pl.sdadas.fstlexicon.command.ParseSentences;
import pl.sdadas.fstlexicon.fst.FSTSearch;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author Sławomir Dadas
 */
public class Application {

    public static void main(String [] args) {
        Map<String, FSTSearch<Long>> dictionaries = new HashMap<>();
        List<Command> commands = createCommands();
        try(BufferedReader reader = new BufferedReader(new InputStreamReader(System.in))) {
            while(true) {
                String line = reader.readLine();
                line = StringEscapeUtils.unescapeJava(line);
                boolean read = false;
                for (Command command : commands) {
                    if(command.readable(line)) {
                        command.read(line, dictionaries);
                        read = true;
                    }
                }
                if(!read) {
                    System.err.println("Unrecognized line: " + StringEscapeUtils.escapeJava(line));
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    private static List<Command> createCommands() {
        List<Command> res = new ArrayList<>();
        res.add(new ParseSentences());
        res.add(new LoadDictionary());
        res.add(new ListDictionaries());
        return res;
    }
}
