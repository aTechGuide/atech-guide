---
title: Smart Contract | Ethereum | BlockChain
description: What is Smart Contract? Smart Contract on Ethereum.
pagetitle: Smart Contract
summary: We will discuss what is Smart Contract. Smart Contract on Ethereum and related Terminology.
date: '2018-06-02'
update_date: '2019-09-09'
tags:
  - BlockChain
label:
  - Blockchain
slug: blockchain/what-is-smart-contract
published: true
image: ../../common/atech-guide.png
---

**Smart Contract**{: .firstword} is a program stored on Blockchain. Function(s) of that program is transparently executed in form of transaction in a completely distributed fashion on blockchain.  
We can think of it as a state Machine which needs transactions to change state

### Characteristics of Smart Contract
- Smart contracts are immutable i.e. once published on network can't be altered.
- We can leverage smart contract to store and update info (Just like OOP). Full history of all the updates is stored on Blockchain.
- Smart Contracts can store balance and hence can be utilized for doing sophisticated value transfers over the internet. Making a way forward to create an **Internet of Value**.
- Multiple Smart contracts can interact with each other to accomplish a complex task.

## Smart Contract on Ethereum
To implement smart contract, we can use Ethereum blockchain. Ethereum uses *proof of work* consensus algorithm. Philosophy behind it states,
> Transactions can have code attach to them which can be run on every node in network

- Ethereum being *Turing Complete* makes it possible to run more sophisticated pieces of logic.
- In Ethereum, We use solidity as high level language to code smart contract which compiles into Ethereum Virtual Machine Assembly Code by Solidity compiler e.g. solcjs. This byte code runs on Ethereum Virtual Machine on every node.
- Compiled bytecode is sent to network which deploys contract and returns an address.(For transaction containing data without a receiver).
- In every 17s there is a competition to decide which node will execute and append a transaction on Ethereum blockchain and winner gets 5 Ethers (which is fixed)
- Research is underway to replace *proof of work* to *proof of stake*
- Smallest unit of currency is 1 Wei

## Terminology
### Opcodes
- EVM bytecode is decoded into series of Ethereum instructions called Opscode
- It is composed of assembly language used by the EVM to execute smart contract

<br/>

### Transaction(s)
- Once code is deployed on network, each function call acts as a transaction changing state on blockchain
- Transactions are processed asynchronously by Miners once sent to network
- Valid transactions are published on blockchain
- Transaction Fields: 
  - *From* (who is executing transaction)
  - *To* (who is recepient, 0 when we deploy contract to chain)
  - *Value* (Optional field indicating how much ether is transfered)
- Each transaction require varying amount of processing power to execute depending on complexity of logic

<br/>

### ABI
- ABI stands for Application Binary interface
- Client doesn't know based on binary of contract on blockchain how to interact with the contract. So it uses ABI which contains all the information of the contract in JSON form.

<br/>

### GAS
- Each transaction carries a certain amount of transaction fee (also known as gas) which we have to pay in crypto currency (in ether for Ethereum) which serves following purposes
  - It acts as an incentive for miners. The miner of block in which this transaction is contained will collect the transaction fee.
  - It makes sure smart contract won't stall network (because of infinite loops etc.). We can associate each transaction with max amount of gas (transaction fee). If the gas ran out before completing transaction then transaction is rolled back but miner still gets the gas.
  - We can raise the priority of transaction by adding more transaction fee
- *Gas Price*: Amount in wei that we are ready to pay per unit of gas.

## Conclusion
Any value that is tracked on a ledger for e.g. results of polling system, copyrights etc can be modeled into a smart contract, leveraging all the benefits of using Blockchain.

## References
- [Getting Started With Ethereum Solidity Development](https://www.udemy.com/getting-started-with-ethereum-solidity-development/){:target="_blank" rel="nofollow" rel="noopener"}
- [Ethereum Masterclass](https://www.udemy.com/ethereum-masterclass/){:target="_blank" rel="nofollow" rel="noopener"}
- [solidity.readthedocs](http://solidity.readthedocs.io/){:target="_blank" rel="nofollow" rel="noopener"}
