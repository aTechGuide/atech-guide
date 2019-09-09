---
title: ICO | ERC20 | BlockChain
description: What is an ICO and ERC20 Token
pagetitle: ICO and ERC20 Token
summary: We will discuss what is an ICO and ERC20 Token.
date: '2018-07-01'
update_date: '2018-07-09'
tags:
  - BlockChain
label:
  - Blockchain
slug: blockchain/what-is-erc20-token
published: true
image: ../../common/atech-guide.png
---

## What is ICO

> **ICO or Initial Coin Offering** means publishing a token smart contract on Ethereum Blockchain. 

<br/>

These are fixed number of tokens which can be bough at an auction, fixed price etc from a fixed date. This token contract has to follow a standard interface which we call *ERC20 Pattern*.

## ERC 20
**ERC**{: .heading1} stands for Ethereum Request for Comments proposal which is a project containing *Ethereum Improvement Proposals*. Its 20th Ethereum request for comments is Standard Token Contract known as ERC20 Contract.  
In nutshell, ERC20 Contract is standard set of functions for,
- Keeping a track of how many tokens each address has
- Transferring tokens from one address to another

### Why to Standardize
Suppose we build our application to interact with an ERC20 Contract. Standardization of ERC20 contract facilitates our application to interact with any ERC20 contract. Hence, we can collect multiple varieties of tokens in our contract just by plugging in the new token contact address without any custom code. 

### Contract Interface

```
 ----------------------------------------------------------------------------
 ERC Token Standard #20 Interface
 https://github.com/ethereum/EIPs/blob/master/EIPS/eip-20-token-standard.md
 ----------------------------------------------------------------------------
contract ERC20Interface {

  string public constant name = "Token Name";
  string public constant symbol = "SYM";
  uint8 public constant decimals = 18;

  function totalSupply() public constant returns (uint);
  function balanceOf(address tokenOwner) public constant returns (uint balance);
  function transfer(address to, uint tokens) public returns (bool success);
  function approve(address spender, uint tokens) public returns (bool success);
  function transferFrom(address from, address to, uint tokens) public returns (bool success);
  function allowance(address tokenOwner, address spender) public constant returns (uint remaining);
  event Transfer(address indexed from, address indexed to, uint tokens);
  event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
}
```
<br/>

**Method Definition**{: .heading1}  

``` function totalSupply() public constant returns (uint); ```  
- Returns Total Supply of Tokens available in Smart Contract.
- Calling this function doesn't cost any Ether

<br/>

``` function balanceOf(address tokenOwner) public constant returns (uint balance); ``` 
- Returns the account balance of account with address ``` tokenOwner ```
- Calling this function doesn't cost any Ether

<br/>

``` function transfer(address to, uint tokens) public returns (bool success); ``` 
- Transfer ``` tokens ``` from Callers address to ```to``` address 
- Fire ``` Transfer ``` event on success.
- Returns True for success

<br/>

``` function approve(address spender, uint tokens) public returns (bool success); ``` 
- We can approve ``` spender ``` to allow him to transfer  ```token``` amount tokens from our address
- Fire ``` Approval ``` event.

<br/>

``` function transferFrom(address from, address to, uint tokens) public returns (bool success); ``` 
- ```spender``` Account can go ahead (once approved) to transfer token from ```from``` address to ```to``` address
- Fire ``` Transfer ``` event.

<br/>

```function allowance(address tokenOwner, address spender) public constant returns (uint remaining);``` 
- Returns amount which ```spender``` is still allowed to withdraw from ```tokenOwner```
- Calling this function doesn't cost any Ether

<br/>

```event Transfer(address indexed from, address indexed to, uint tokens);``` 
- Fire when tokens are successfully transferred.

<br/>

```event Approval(address indexed tokenOwner, address indexed spender, uint tokens);``` 
- Fire on successful call to approve.

<br/>

**ERC20 Token Variables**{: .heading1}  

- ```string public constant name = "Token Name";``` Name of Token
- ```string public constant symbol = "SYM";``` Symbol of Token used by exchanges
- ```uint8 public constant decimals = 18;``` Decimals give us room to spend fraction of a token.

## References 
- [Ethereum Masterclass](https://www.udemy.com/ethereum-masterclass/){:target="_blank" rel="nofollow" rel="noopener"}
- [ERC20_Token_Standard](https://theethereum.wiki/w/index.php/ERC20_Token_Standard){:target="_blank" rel="nofollow" rel="noopener"}
- [GitHub EIPS/eip-20](https://github.com/ethereum/EIPs/blob/master/EIPS/eip-20.md){:target="_blank" rel="nofollow" rel="noopener"}
- [Cryptozombies.io](https://cryptozombies.io/){:target="_blank" rel="nofollow" rel="noopener"}
