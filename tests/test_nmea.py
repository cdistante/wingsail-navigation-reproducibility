"""Checks for the OpenCPN NMEA 0183 bridge."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from world.nmea import (
    NMEA0183TCPClientOutput,
    NMEA0183TCPServerOutput,
    NMEA0183UDPOutput,
    hdt_sentence,
    nmea_checksum,
    rmc_sentence,
)


class NMEATests(unittest.TestCase):
    """Ensure the simulator produces standards-shaped navigation messages."""

    def test_rmc_contains_simulated_position_speed_course_and_checksum(self) -> None:
        sentence = rmc_sentence(
            np.datetime64("2026-07-26T21:01:11.250", "ns"),
            39.88,
            18.42,
            1.0,
            49.0,
        )

        self.assertTrue(sentence.startswith("$GPRMC,210111.25,A,3952.8000,N,01825.2000,E,1.94,49.0,260726"))
        body, checksum = sentence[1:].strip().split("*")
        self.assertEqual(int(checksum, 16), nmea_checksum(body))

    def test_hdt_uses_true_heading(self) -> None:
        self.assertTrue(hdt_sentence(361.5).startswith("$HCHDT,1.5,T*"))

    def test_udp_output_delivers_complete_sentences(self) -> None:
        sentences = (hdt_sentence(90.0), hdt_sentence(180.0))
        with patch("world.nmea.socket.socket") as socket_factory:
            with NMEA0183UDPOutput("127.0.0.1", 10110) as output:
                output.send(*sentences)

        self.assertEqual(
            socket_factory.return_value.sendto.call_args_list,
            [
                unittest.mock.call(sentence.encode("ascii"), ("127.0.0.1", 10110))
                for sentence in sentences
            ],
        )
        socket_factory.return_value.close.assert_called_once()

    def test_tcp_client_output_sends_complete_sentences(self) -> None:
        sentence = hdt_sentence(90.0)
        with patch("world.nmea.socket.create_connection") as connect:
            with NMEA0183TCPClientOutput("127.0.0.1", 3000) as output:
                output.send(sentence)

        connect.assert_called_once_with(("127.0.0.1", 3000))
        connect.return_value.sendall.assert_called_once_with(sentence.encode("ascii"))
        connect.return_value.close.assert_called_once()

    def test_tcp_server_output_accepts_and_sends_to_opencpn(self) -> None:
        sentence = hdt_sentence(90.0)
        with patch("world.nmea.socket.socket") as socket_factory:
            listener = socket_factory.return_value
            connection = MagicMock()
            listener.accept.return_value = (connection, ("127.0.0.1", 50000))
            with NMEA0183TCPServerOutput("127.0.0.1", 3000) as output:
                output.send(sentence)

        listener.bind.assert_called_once_with(("127.0.0.1", 3000))
        listener.listen.assert_called_once_with(1)
        connection.sendall.assert_called_once_with(sentence.encode("ascii"))
        connection.close.assert_called_once()
        listener.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
