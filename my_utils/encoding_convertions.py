import re
import numpy as np

ENCODING_OPTIONS = ["kern", "decoupled", "decoupled_dot"]


class krnConverter:
    """Main Kern converter operations class."""

    def __init__(self, encoding: str = "kern", keep_ligatures: bool = True):
        self.reserved_words = ["clef", "k[", "*M"]
        self.reserved_dot = "."
        self.reserved_dot_EncodedCharacter = "DOT"
        self.clef_change_other_voices = "*"
        self.comment_symbols = ["*", "!"]
        self.voice_change = "\t"
        self.step_change = "\n"
        self.keep_ligatures = keep_ligatures

        # Convert function
        assert (
            encoding in ENCODING_OPTIONS
        ), f"You must chose one of the possible encoding options: {','.join(ENCODING_OPTIONS)}"
        self.encoding = encoding
        self.convert_function_options = {
            "kern": self.cleanKernFile,
            "decoupled": self.cleanAndDecoupleKernFile,
            "decoupled_dot": self.cleanAndDecoupleDottedKernFile,
        }
        self.convert_step = self.convert_function_options[self.encoding]

    def _readSrcFile(self, file_path: str) -> list:
        """Read polyphonic kern file and adequate the format for further processes."""
        with open(file_path) as fin:
            in_src = fin.read().splitlines()

        # Locating line with the headers:
        it_headers = 0
        while "**kern" not in in_src[it_headers]:
            it_headers += 1
        pass
        columns_to_process = np.where(
            np.array(in_src[it_headers].split("\t")) == "**kern"
        )[0]

        # Locating lines with comments (to be removed):
        in_src_nocomments = list()
        for line in in_src:
            if not line.strip().startswith("!"):
                in_src_nocomments.append(line.split("\t"))
        pass

        # Extract voices and removing lines with comments:
        out_src = np.array(in_src_nocomments)[:, columns_to_process]

        return out_src

    def _postprocessKernSequence(self, in_score: list) -> list:
        """Exchanging '*' for the actual symbol; removing ligatures (if so)"""

        # Retrieving positions with '*':
        positions = sorted(list(set(np.where(in_score == "*")[1])))

        # For each position, we retrieve the last explicit clef symbol and include it in the stream:
        for single_position in positions:
            for it_voice in range(in_score.shape[0]):
                if in_score[it_voice, single_position] == "*":
                    new_element = in_score[
                        it_voice,
                        max(
                            np.where(np.char.startswith(in_score[it_voice], "*clef"))[0]
                        ),
                    ]
                    in_score[it_voice, single_position] = new_element
                pass
            pass
        pass

        # Removing ligatures (if so):
        ### Locating openings:
        if not self.keep_ligatures:
            for it_voice in range(in_score.shape[0]):
                for pos in np.where(np.char.startswith(in_score[it_voice], "["))[0]:
                    in_score[it_voice][pos] = in_score[it_voice][pos].replace("[", "")
                pass

                for pos in np.where(np.char.endswith(in_score[it_voice], "]"))[0]:
                    if not np.char.startswith(in_score[it_voice][pos], "*"):
                        in_score[it_voice][pos] = in_score[it_voice][pos].replace(
                            "]", ""
                        )
                    pass
                pass
            pass
        pass

        return in_score

    def cleanKernFile(self, file_path: str) -> list:
        """Convert complete kern sequence to CLEAN kern format."""
        in_file = self._readSrcFile(file_path=file_path)

        # Processing individual voices:
        out_score = list()
        for it_voice in range(in_file.shape[1]):
            in_voice = in_file[:, it_voice].tolist()
            out_voice = [
                self.cleanKernToken(u)
                for u in in_voice
                if self.cleanKernToken(u) is not None
            ]

            out_score.append(out_voice)
        pass
        out_score = np.array(out_score)

        # Postprocess obtained score:
        out_score = self._postprocessKernSequence(out_score)

        return out_score

    def cleanKernToken(self, in_token: str) -> str:
        """Convert a kern token to its CLEAN equivalent."""
        out_token = None  # Default

        if any(
            [u in in_token for u in self.reserved_words]
        ):  # Relevant reserved tokens
            out_token = in_token

        elif in_token == self.reserved_dot:  # Case when using "." for sync. voices
            out_token = self.reserved_dot_EncodedCharacter

        elif (
            in_token.strip() == self.clef_change_other_voices
        ):  # Clef change in other voices
            out_token = in_token

        elif any([in_token.startswith(u) for u in self.comment_symbols]):  # Comments
            out_token = None

        elif in_token.startswith("s"):  # Slurs
            out_token = "s"

        elif "=" in in_token:  # Bar lines
            out_token = "="

        elif not "q" in in_token:
            if "rr" in in_token:  # Multirest
                out_token = re.findall("rr[0-9]+", in_token)[0]
            elif "r" in in_token:  # Rest
                out_token = in_token.split("r")[0] + "r"
            else:  # Music note
                out_token = re.findall("\[*\d+[.]*[a-gA-G]+[n#-]*\]*", in_token)[0]

        return out_token

    # ---------------------------------------------------------------------------- DECOUPLE

    def _postprocessDecoupleSequence(self, in_score: list) -> list:
        """Exchanging '*' for the actual symbol."""

        # Retrieving positions with '*':
        positions = sorted(list(set(np.where(in_score == "*")[1])))

        # For each position, we retrieve the last explicit clef symbol and include it in the stream:
        for single_position in positions:
            for it_voice in range(in_score.shape[0]):
                if in_score[it_voice, single_position] == "*":
                    new_element = in_score[
                        it_voice,
                        max(
                            np.where(np.char.startswith(in_score[it_voice], "*clef"))[0]
                        ),
                    ]
                    in_score[it_voice, single_position] = new_element
                pass
            pass
        pass

        return in_score

    def _normalizingLengthCleanAndDecoupledSequences(self, in_voices: list) -> list:
        """Normalizing the length of the individual voices."""

        for it_step in range(len(in_voices[0])):
            # Obtaining individual lengths per time step (only symbols and avoiding None elements):
            lengths = [
                len(np.where(np.array(in_voices[it_voice][it_step]) != None)[0])
                for it_voice in range(len(in_voices))
            ]

            # Adding "self.reserved_dot_EncodedCharacter" symbols to compensate lengths:
            if len(set(lengths)) > 1:
                max_length = max(lengths)
                for it_voice in range(len(in_voices)):
                    if lengths[it_voice] < max_length:
                        in_voices[it_voice][it_step].extend(
                            [
                                self.reserved_dot_EncodedCharacter
                                for _ in range(max_length - lengths[it_voice])
                            ]
                        )
                    pass
                pass
            pass
        pass

        return in_voices

    def cleanAndDecoupleKernFile(self, file_path: str) -> list:
        """Convert complete kern sequence to CLEAN and DECOUPLED kern format."""
        in_file = self._readSrcFile(file_path=file_path)

        # Processing individual voices:
        temp_out_score = list()
        for it_voice in range(in_file.shape[1]):
            in_voice = in_file[:, it_voice].tolist()
            out_voice = [self.cleanAndDecoupleKernToken(u) for u in in_voice]
            temp_out_score.append(out_voice)
        pass

        # Normalizing the decomposition of the individual symbols:
        temp_out_score = self._normalizingLengthCleanAndDecoupledSequences(
            temp_out_score
        )

        # Removing 'None' symbols and flattening the individual voices:
        out_score = list()
        for it_voice in range(len(temp_out_score)):
            out_voice = [
                x for xs in temp_out_score[it_voice] for x in xs if x is not None
            ]
            out_score.append(out_voice)
        pass

        # To numpy array:
        out_score = np.array(out_score)

        # Processing clef changes:
        out_score = self._postprocessDecoupleSequence(out_score)

        return out_score

    def cleanAndDecoupleKernToken(self, in_token: str) -> list:
        """Convert a kern token to its CLEAN and DECOUPLED equivalent."""
        out_token = []  # Default

        if any(
            [u in in_token for u in self.reserved_words]
        ):  # Relevant reserved tokens
            out_token.append(in_token)

        elif in_token == self.reserved_dot:  # Case when using "." for sync. voices
            out_token = [self.reserved_dot_EncodedCharacter]

        elif (
            in_token.strip() == self.clef_change_other_voices
        ):  # Clef change in other voices
            out_token = in_token

        elif any([in_token.startswith(u) for u in self.comment_symbols]):  # Comments
            out_token.append(None)

        elif in_token.startswith("s"):  # Slurs
            out_token.append("s")

        elif "=" in in_token:  # Bar lines
            out_token.append("=")

        elif not "q" in in_token:
            if "rr" in in_token:  # Multirest
                out_token.append(re.findall("rr[0-9]+", in_token)[0])
            elif "r" in in_token:  # Rest
                out_token = [in_token.split("r")[0], "r"]
            else:  # Notes
                # Extract opening ligature (if exists):
                rest_note = in_token
                if "[" in in_token:
                    ligature = "["
                    rest_note = in_token.split("[")[1]
                    if self.keep_ligatures:
                        out_token.append(ligature)

                # Extract duration:
                duration = re.findall("\d+", rest_note)[0]
                rest = re.split("\d+", rest_note)[1]
                out_token.append(duration)

                # Extract dot (if exists):
                dot = [None]
                if "." in rest:
                    dot = list(re.findall("[.]+", rest)[0])
                    rest = re.split("[.]+", rest)[1]
                out_token.extend(dot)

                # Extract pitch:
                pitch = re.findall("[a-gA-G]+", rest)[0]
                rest = re.split("[a-gA-G]+", rest)[1]
                out_token.append(pitch)

                # Extract alteration (if exists):
                alteration = None
                alteration = re.findall("[n#-]*", rest)[0]
                if len(alteration) == 0:
                    alteration = None
                out_token.append(alteration)

                # Extract closing ligature (if exists):
                if "]" in rest:
                    ligature = "]"
                    rest_note = in_token.split("]")[1]
                    if self.keep_ligatures:
                        out_token.append(ligature)
                pass
            pass

        else:
            out_token = [None]

        return out_token

    # ---------------------------------------------------------------------------- DECOUPLE DOTTED

    def cleanAndDecoupleDottedKernFile(self, file_path: str) -> list:
        """Convert complete kern sequence to CLEAN and DECOUPLED kern format."""
        in_file = self._readSrcFile(file_path=file_path)

        # Processing individual voices:
        temp_out_score = list()
        for it_voice in range(in_file.shape[1]):
            in_voice = in_file[:, it_voice].tolist()
            out_voice = [self.cleanAndDecoupleDottedKernToken(u) for u in in_voice]
            temp_out_score.append(out_voice)
        pass

        # Normalizing the decomposition of the individual symbols:
        temp_out_score = self._normalizingLengthCleanAndDecoupledSequences(
            temp_out_score
        )

        # Removing 'None' symbols and flattening the individual voices:
        out_score = list()
        for it_voice in range(len(temp_out_score)):
            out_voice = [
                x for xs in temp_out_score[it_voice] for x in xs if x is not None
            ]
            out_score.append(out_voice)
        pass

        # To numpy array:
        out_score = np.array(out_score)

        # Processing clef changes:
        out_score = self._postprocessDecoupleSequence(out_score)

        return out_score

    def cleanAndDecoupleDottedKernToken(self, in_token: str) -> list:
        """Convert a kern token to its CLEAN and DECOUPLED-with-DOT equivalent."""
        out_token = []  # Default

        if any(
            [u in in_token for u in self.reserved_words]
        ):  # Relevant reserved tokens
            out_token.append(in_token)

        elif in_token == self.reserved_dot:  # Case when using "." for sync. voices
            out_token = [self.reserved_dot_EncodedCharacter]

        elif (
            in_token.strip() == self.clef_change_other_voices
        ):  # Clef change in other voices
            out_token = in_token

        elif any([in_token.startswith(u) for u in self.comment_symbols]):  # Comments
            out_token.append(None)

        elif in_token.startswith("s"):  # Slurs
            out_token.append("s")

        elif "=" in in_token:  # Bar lines
            out_token.append("=")

        elif not "q" in in_token:
            if "rr" in in_token:  # Multirest
                out_token.append(re.findall("rr[0-9]+", in_token)[0])
            elif "r" in in_token:  # Rest
                out_token = [in_token.split("r")[0], "r"]
            else:  # Notes
                # Extract opening ligature (if exists):
                rest_note = in_token
                if "[" in in_token:
                    ligature = "["
                    rest_note = in_token.split("[")[1]
                    if self.keep_ligatures:
                        out_token.append(ligature)

                # Extract duration:
                duration = re.findall("\d+", rest_note)[0]
                rest = re.split("\d+", rest_note)[1]

                # Extract dot (if exists)
                if "." in rest:
                    dot = list(re.findall("[.]+", rest)[0])
                    duration += "".join(dot)
                    rest = re.split("[.]+", rest)[1]
                out_token.append(duration)

                # Extract pitch:
                pitch = re.findall("[a-gA-G]+", rest)[0]
                rest = re.split("[a-gA-G]+", rest)[1]
                out_token.append(pitch)

                # Extract alteration (if exists):
                alteration = None
                alteration = re.findall("[n#-]*", rest)[0]
                if len(alteration) == 0:
                    alteration = None
                out_token.append(alteration)

                # Extract closing ligature (if exists):
                if "]" in rest:
                    ligature = "]"
                    rest_note = in_token.split("]")[1]
                    if self.keep_ligatures:
                        out_token.append(ligature)
                pass
            pass

        else:
            out_token = [None]

        return out_token

    # ---------------------------------------------------------------------------- CONVERT CALL

    def convert(self, src_file: str) -> list:
        out = self.convert_step(src_file).T

        out_line = self.step_change.join(
            [self.voice_change.join(voice) for voice in out]
        )

        return out_line
