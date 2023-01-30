# FL_demo

## Intro

The project is a basic implementation of the classical and split Federated Learning (FL) system that uses a traditional centralized architecture, transmits model parameters in plaintext, and is implemented using multithreading simulation.

Dataset: MNIST, FASHION-MNIST

Network: DNN， `MnistNet` in `plain_network.py`

Control:  `config.py`。

## Dependency libraries

Python >= 3.7

pytorch >= 1.8

## Optional protocol

**ALL**: classical FL protocol, local models **fully** replicates global parameters, supports the number of clients from 2 to 100

**SEP**: split FL protocol, local models only updates **a portion of** the global parameters, supports only 4 clients now

## Communication

Each Client holds three sockets for communication：

`tcp_client_socket_server`: TCP socket, used to send and receive data from the Server

`udp_send_socket`: UDP socket, used to send data to other Client

`udp_recv_socket`: UDP socket, used to receive data from other Client

The Server only needs a TCP socket

### Initial phase

1. Each Client uploads the self information to Server.
2. The Server collects information about all Clients, establishes a communication *lookup table*, and broadcasts it.
   lookup table: [client id, UDP send port, UDP recv port]
3. The Clients perform internal communication based on the contents of the lookup table.

### Learning phase

1. Each Client uploads the local parameters to Server.
2. Server runs the aggregation by *FedAvg*[[1]](McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data[C]//Artificial intelligence and statistics. PMLR, 2017: 1273-1282.) and broadcast the global parameters.  

## Example

To run the system, do:

```
python main.py
```

The terminal shows the test loss and accuracy for each epoch. And the loss and accuracy and other information for each batch are recorded in the `logs` folder.。

## Other problems

- If an error occurs：OSError: [Errno 98] Address already in use

  The cause is that the server port is occupied. Please change the **SERVER_PORT** to any free port in `train_params.py`.
