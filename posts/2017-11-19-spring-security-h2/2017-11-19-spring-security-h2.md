---
title: Spring Security with H2
description: We will learn how to configure h2 db with spring security and access its console.
pagetitle: Spring Boot Security with H2
summary: A simple secure spring web application needs a database to be configures at its backend. This post covers step by step how to configure h2 db and access its console.
date: '2017-11-19'
update_date: '2019-09-08'
tags:
  - Spring
  - SpringSecurity
label:
  - Spring
slug: spring-security-h2
published: true
image: ../common/atech-guide.png
---

## Introduction 

A simple secure spring web application needs a database to be configures at its backend. This post covers how to configure h2 db and access its console.

### Maven dependency

To add H2 in spring boot application, you need to add following maven dependency in pom.xml

```java
<dependency>
    <groupId>com.h2database</groupId>
    <artifactId>h2</artifactId>
</dependency>
```

Spring boot will automatically configure it when it sees its dependency on classpath.

### To access the console. 
You need to add the following bean.

```java
@Configuration
public class H2Config {

    @Bean
    ServletRegistrationBean h2servletRegistration(){
        ServletRegistrationBean registrationBean = new ServletRegistrationBean( new WebServlet());
        registrationBean.addUrlMappings("/console/*");
        return registrationBean;
    }
}
```

That's it, fire up your application and try hitting URL: http://localhost:8080/console/  

It will prompt for username and password, which will be  
**username** : user  
**password**: <check for log line, "Using default security password: 3cbb1572-36c6-4487-b602-11ab2dc576ad">  

### To access the console without login 
Add the following security configuration

```java
@Configuration
public class SecurityConfig extends WebSecurityConfigurerAdapter {

    @Override
    protected void configure(HttpSecurity httpSecurity) throws Exception {

        httpSecurity
                .authorizeRequests().antMatchers("/console/**").permitAll()
                .anyRequest().authenticated();

        httpSecurity.csrf().disable();
        httpSecurity.headers().frameOptions().disable();
    }
}
```

You're good to go.   
To access the full working code sample, click [here](https://github.com/kamranalinitb/springboot-blog/tree/master/security-h2){:target="_blank" rel="nofollow"}

### References
- [Spring Frameworkguru](https://springframework.guru/using-the-h2-database-console-in-spring-boot-with-spring-security/){:target="_blank" rel="nofollow"}
