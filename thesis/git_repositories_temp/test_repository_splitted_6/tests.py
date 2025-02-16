# encoding: UTF-8
# Copyright 2017 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import unittest
import my_txtutils as txt

TST_TXTSIZE = 10000
TST_SEQLEN = 10
TST_BATCHSIZE = 13
TST_EPOCHS = 5


class RnnMinibatchSequencerTest(unittest.TestCase):
    def setUp(self):
        # generate text of consecutive items
        self.data = list(range(TST_TXTSIZE))

    @staticmethod
    def check_seq_batch(batch1, batch2):
        nb_errors = 0
        for i in range(TST_BATCHSIZE):
            ok = batch1[i, -1] + 1 == batch2[i, 0]
            nb_errors += 0 if ok else 1
        return nb_errors

    def test_sequences(self):
        for x, y, epoch in txt.rnn_minibatch_sequencer(self.data, TST_BATCHSIZE, TST_SEQLEN, TST_EPOCHS):
            for i in range(TST_BATCHSIZE):
                self.assertListEqual(x[i, 1:].tolist(), y[i, :-1].tolist(),
                                     msg="y sequences must be equal to x sequences shifted by -1")

    def test_batches(self):
        start = True
        prev_x = np.zeros([TST_BATCHSIZE, TST_SEQLEN], np.int32)
        prev_y = np.zeros([TST_BATCHSIZE, TST_SEQLEN], np.int32)
        nb_errors = 0
        nb_batches = 0
        for x, y, epoch in txt.rnn_minibatch_sequencer(self.data, TST_BATCHSIZE, TST_SEQLEN, TST_EPOCHS):
            if not start:
                nb_errors += self.check_seq_batch(prev_x, x)
                nb_errors += self.check_seq_batch(prev_y, y)
            prev_x = x
            prev_y = y
            start = False
            nb_batches += 1
        self.assertLessEqual(nb_errors, 2 * TST_EPOCHS,
                             msg="Sequences should be correctly continued, even between epochs. Only "
                                 "one sequence is allowed to not continue from one epoch to the next.")
        self.assertLess(TST_TXTSIZE - (nb_batches * TST_BATCHSIZE * TST_SEQLEN),
                        TST_BATCHSIZE * TST_SEQLEN * TST_EPOCHS,
                        msg="Text ignored at the end of an epoch must be smaller than one batch of sequences")


class EncodingTest(unittest.TestCase):
    def setUp(self):
        self.test_text_known_chars = \
            "PRIDE AND PREJUDICE"
        self.test_text_unknown_char = "Unknown char: \x0C"  # the unknown char 'new page'

    def test_encoding(self):
        encoded = txt.encode_text(self.test_text_known_chars)
        decoded = txt.decode_to_text(encoded)
        self.assertEqual(self.test_text_known_chars, decoded,
                         msg="On a sequence of supported characters, encoding, "
                             "then decoding should yield the original string.")

    def test_unknown_encoding(self):
        encoded = txt.encode_text(self.test_text_unknown_char)
        decoded = txt.decode_to_text(encoded)
        original_fix = self.test_text_unknown_char[:-1] + chr(0)
        self.assertEqual(original_fix, decoded,
                         msg="The last character of the test sequence is an unsupported "
                             "character and should be encoded and decoded as 0.")


class TxtProgressTest(unittest.TestCase):
    def test_progress_indicator(self):
        print("If the printed output of this test is incorrect, the test will fail. No need to check visually.", end='')
        test_cases = (50, 51, 49, 1, 2, 3, 1000, 333, 101)
        p = txt.Progress(100)
        for maxi in test_cases:
            m, cent = self.check_progress_indicator(p, maxi)
            self.assertEqual(m, maxi, msg="Incorrect number of steps.")
            self.assertEqual(cent, 100, msg="Incorrect number of steps.")

    @staticmethod
    def check_progress_indicator(p, maxi):
        p._Progress__print_header()
        progress = p._Progress__start_progress(maxi)
        total = 0
        n = 0
        for k in progress():
            total += k
            n += 1
        return n, total
