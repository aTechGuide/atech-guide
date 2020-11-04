---
title: Spring JWT | Parts of JWT
description: We will learn what is JWT? Parts of JWT? Spring Integration of JWT.
pagetitle: Spring Boot JWT
summary: JWT is a tokenization format in which we pass our credentials and content in a single format wrapped inside token.
date: '2017-12-12'
update_date: '2019-09-08'
tags:
  - Spring
  - SpringSecurity
label:
  - Spring
slug: springjwt
published: true
image: ../../common/atech-guide.png
---

## What is JWT

As per JWT.io  
> JSON Web Tokens are an open, industry standard RFC 7519 method for representing claims securely between two parties.

It is a tokenization format in which we pass our credentials and content in a single format wrapped inside token.  

### Parts of JWT

- **Header**{: .heading1}
  - Contains info of Hashing methodology we have used to encode our message
  - Algorithm and Token Type info
- **Payload**{: .heading1}
  - Data/Message to be transferred
- **Signature**{: .heading1}
  - Encoded Header and Payload then merge them along with a secret
  
## Spring Integration

To access the full working code sample, click [here](https://github.com/atechguide/springboot-blog/tree/master/jwt){:target="_blank" rel="nofollow" rel="noopener"}

## Reference
- [What is JWT? JWT Vs OAuth | Tech Primers](https://www.youtube.com/watch?v=muRr4dImv1k){:target="_blank" rel="nofollow"}  
- [JWT.io](https://jwt.io/){:target="_blank" rel="nofollow" rel="noopener"}  
- [Spring Security using JWT in Spring Boot App | Tech Primers](https://www.youtube.com/watch?v=-HYrUs1ZCLI){:target="_blank" rel="nofollow" rel="noopener"}
