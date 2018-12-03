package pl.sdadas.fstlexicon.command;

import org.apache.commons.lang3.StringEscapeUtils;
import org.apache.commons.lang3.StringUtils;
import pl.sdadas.fstlexicon.fst.FSTMatch;
import pl.sdadas.fstlexicon.fst.FSTSearch;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.Map;

/**
 * @author Sławomir Dadas
 */
public class ParseSentence implements Command {

    @Override
    public boolean readable(String input) {
        return StringUtils.startsWith(input.trim(), "--parse");
    }

    @Override
    public void read(String input, Map<String, FSTSearch<Long>> dictionaries) {
        String line = input.trim();
        String[] args = StringUtils.split(line, " ", 3);
        if(args.length < 3) {
            String msg = "Cannot parse line, format for parsing should be '--parse dictionary_name base64_sentence'";
            throw new IllegalStateException(msg);
        }
        String name = StringUtils.lowerCase(args[1]);
        FSTSearch<Long> fst = dictionaries.get(name);
        if(fst == null) {
            String msg = String.format("Dictionary '%s' not found", name);
            throw new IllegalStateException(msg);
        }
        String output = findMatches(args[2], fst);
        System.out.println(StringEscapeUtils.escapeJava(output));
    }

    private String findMatches(String base64, FSTSearch<Long> fst) {
        byte[] bytes = Base64.getDecoder().decode(base64);
        String decoded = new String(bytes, StandardCharsets.UTF_8);
        String[] inputWords = StringUtils.split(decoded, "\n");
        List<FSTMatch<Long>> matches = fst.find(inputWords);
        String[] outputWords = new String[inputWords.length];
        Arrays.fill(outputWords, "");
        matches.forEach(match -> applyMatch(match, outputWords));
        String output = StringUtils.join(outputWords, '\n');
        return Base64.getEncoder().encodeToString(output.getBytes(StandardCharsets.UTF_8));
    }

    private void applyMatch(FSTMatch<Long> match, String[] output) {
        String val = match.getValue().toString();
        for (int i = match.getStartIdx(); i <= match.getEndIdx(); i++) {
            output[i] = val;
        }
    }
}