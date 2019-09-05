---
title: What are Reactive Systems and Reactive Programming. 4 Characteristics of Reactive Systems.
description: What is Reactive Programming. Properties of Reactive Systems
pagetitle: Reactive Programming and Reactive Systems
summary: What is Reactive Programming and properties of Reactive Systems
logo: fas fa-stream
date: '2017-08-10'
update_date: '2017-08-10'
tags:
  - Technology
slug: reactive-systems-and-reactive-programming
published: false
---

## What are Reactive Systems?

> Reactive Systems in a nutshell is an Architectural and Design pattern of building large scale, self healing systems e.g. Googles' Gmail, Amazon.com etc

<br/>

Based on guidelines provided by [Reactive Manifesto](https://www.reactivemanifesto.org/){:target="_blank" rel="nofollow noopener"}, reactive systems possess following characteritics:
- **Responsiveness:** 
  - Reactive Systems should respond in *consistent and timely manner.* i.e. when an API call is made we expect a predictable response within SLAs.
  - Reactive Systems will detect problems quickly and mitigate it effectively.
    - It demonstrates the automated self healing nature of reactive systems in which once bleeding begins it quickly detects it and take appropriate measures to mitigate it
- **Resilient:** 
  - Reactive Systems should be responsive in face of failure.
    - Failure(s) in one business flow doesn't bring the entire system to a grinding halt.
  - Resilience is achieved by 
    - *Replication:* To ensure high availability we use replication. e.g. running cluster of Redis Server.
    - *Containment:* Failures are contained within each component. e.g. failures in connecting to one downstream doesn't stops entire system
    - *Isolation:* Components of a system are isolated ensuring part of system can fail and recover without compromising the entire system.
    - *Delegation:* Recovery of each component is delegated to another component (maybe external). e.g. falling back to alternate data source if primary data source is unavailable
- **Elastic:** 
  - Reactive Systems should be responsive under varying workload.
  - Reactive Systems react to changes in the input rate by increasing or decreasing the resources allocated to service these inputs. So when load increases additional nodes are added and vice versa.
  - Elasticity is achieved in cost-effective way on commodity hardware and software platforms.
- **Message Driven:** 
  - Reactive systems have *Asynchronous Messaging* i.e. while requesting data we don’t wait for response, instead we register a callback and return. When data is available, it is pushed to callback method.
    - Asynchronous Messaging establishes a boundary between components ensuring 
      - *Loose coupling*
      - *Isolation*
      - *Location transparency* implying we are not dependent on server A talking to server B instead a cluster of micro service 'A' talks to cluster of micro service 'B'.
  - Message driven communication between system ensures
    - Failures are notified in form of messages
    - Proper Load Management
    - Elasticity
  - Non-blocking communication facilitates only consume resources while active, leading to less system overhead.

## What is Reactive Programming?

> Reactive Programming is an Asynchronous Non-blocking programming paradigm to process streams of data. Suppose we have a stream of stock trades and the programming that is reacting on those events is called *Reactive Programming*. Its a tool of building Reactive Systems.

### Salient Features of Reactive Programming
The nuts and bolts of Reactive Programming can be categories as follows,
- Data Streams
  - Stream is a sequence of events ordered in time e.g. Twitter feed, Stock trades etc
- Asynchronous
  - Events are captured Asynchronously 
  - Callback function is defined to execute 
    - when an event is emitted
    - when an error is emitted
    - when complete event is emitted
- Non-blocking
  - Non Blocking means we will process the available data (from disk, network etc) and will return. In contrast, with blocking code which will stop and wait for more data till eof is reached or network call is completed.
- Backpreassure
  - Subscriber is able to throttle data i.e. gimme 10 records. Basically facilitating a communication between subscriber and publisher
- Failures as messages 
  - We tread failures as *First Class Citizens*.
  - In processing streams of data if we threw an exception we will break the processing of stream so instead we will gracefully handle the exceptions through handler functions

### Implementations of Reactive Programming
There are multiple libraries available which implements reactive programming in multiple language. Comprehensive list can be found [here](https://xgrommx.github.io/rx-book/content/resources/reactive_libraries/index.html){:target="_blank" rel="nofollow noopener"}
Reactive programming can be understood by relating it to observable design Pattern in which
- Subject will notify the observer for changes
  - ReactiveX Observables
    - add from http://reactivex.io/documentation/observable.html


## Implementations
Netflix has implemented Reactive Programming paradigm and provided us with a [RxJava](https://github.com/ReactiveX/RxJava){:target="_blank" rel="nofollow noopener"} library which works on the principle of Observable Design Pattern. Its official document states,
> RxJava – Reactive Extensions for the JVM – a library for composing asynchronous and event-based programs using observable sequences for the Java VM.

## Observable Design Pattern
To understand Observable Design Pattern, lets juxtapose it over iterables, 
- **Iterables**: are used when we want to pull data from list or collection.
- **Observable**: are used when data is pushed to clients from server side.

## Reference
- [Reactive manifesto](https://www.reactivemanifesto.org/){:target="_blank" rel="nofollow noopener"}
- [What is Reactive Programming? - Tech Primers](https://www.youtube.com/watch?v=0ueFTvSdxpw){:target="_blank" rel="nofollow noopener"}
- [RxJava](https://github.com/ReactiveX/RxJava){:target="_blank" rel="nofollow noopener"}
- [Reactivex.io - observable](http://reactivex.io/documentation/observable.html){:target="_blank" rel="nofollow noopener"}