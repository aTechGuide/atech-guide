---
title: Private Network | Geth | BlockChain
description: How to start private network using Geth
pagetitle: Creating Private Network
summary: We will start our own private network using Geth (GO Ethereum)
date: '2018-06-30'
update_date: '2019-09-09'
tags:
  - BlockChain
label:
  - Blockchain
slug: blockchain/start-geth-private-network
published: true
image: ../../common/atech-guide.png
---

**Private network**{: .firstword} is the full fledged Ethereum blockchain created on our computer. We can use Geth (Go Ethereum Client) to start our own private network on our computer.

## Requirements
### Install Geth 

Geth is a cli that allows us to run and operate full Etherium node. On Mac we can install it using following commands

```
brew tap ethereum/ethereum
brew install ethereum
```

### Creating Genesis Block
Genesis Block is the *First Block* in a Blockchian.
- Creating genesis block and associating it with a network is called *Instantiation*
- Chain of blocks that build on top of this block is called *Instance*
- Genesis block + Network identifier, is all we need to create a totally different history line

To create Genesis Block we need a Genesis file which is a JSON file, like following

```
{
  "coinbase"   : "0x0000000000000000000000000000000000000001",
  "difficulty" : "0x10000",
  "extraData"  : "",
  "gasLimit"   : "0x9000000",
  "nonce"      : "0x0000000000000032",
  "mixhash"    : "0x0000000000000000000000000000000000000000000000000000000000000000",
  "parentHash" : "0x0000000000000000000000000000000000000000000000000000000000000000",
  "timestamp"  : "0x00",
  "alloc": {},
  "config": {
        "chainId": 15,
        "homesteadBlock": 0,
        "eip155Block": 0,
        "eip158Block": 0
    }
}
```

## Steps
- Create an empty folder ``` privateNetwork ```
- Place ``` genesis.json ``` file into it
- Create an empty folder ``` chaindata ```
- Initialize the private chain ``` geth --datadir=./chaindata init ./genesis.json ```  
We will see following lines as part of output

```
...
...
Allocated cache and file handles  database=/your/directory/privateNetwork/chaindata/geth/chaindata cache=16 handles=16
...
Successfully wrote genesis state  database=lightchaindata      hash=9b8d4a…9021ba
```

- Start Private Network ``` geth --datadir=./chaindata ```  

We will see following lines as part of output

```
...
...
Initialised chain configuration  config="{ChainID: 15 Homestead: 0 DAO: <nil> DAOSupport: false EIP150: <nil> EIP155: 0 EIP158: 0 Byzantium: <nil> Engine: unknown}"
...
...
```

You will notice ``` ChainID: 15 ``` is the same ChainID that we have specified in genesis.json file. ``` ChainID: 1 ``` is main Ethereum network.

## JSON RPC Interface of Geth
To attach to JSON RPC interface of Geth, We can use following command
- To start Geth and attach to its RPC Console ``` geth --datadir=./chaindata console ```
- To attach to already started Geth ``` geth --datadir=./chaindata attach ``` OR ``` geth attach ipc:/path/to/ipc/file/geth.ipc ```

## To start HTTP Restful Interface of Geth
To communicate with Geth from outside Geth console, we need to start Geth with Restful Interface. We use ``` --rpc ``` parameter which will open port ``` 8545 ``` on localhost. We also need to enable CORS (Cross Origin Resource Sharing) by ``` --rpccorsdomain "*" ```. Once it is opened we will see following line in output. 

```
...
...
HTTP endpoint opened                     url=http://127.0.0.1:8545
...
```

## References
- [Ethereum Masterclass](https://www.udemy.com/course/ethereum-masterclass/){:target="_blank" rel="nofollow" rel="noopener"}
