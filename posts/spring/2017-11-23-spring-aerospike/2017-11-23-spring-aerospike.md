---
title: Spring Aerospike
description: We will learn what is Aerospike? what are layers in Aerospike? Aerospike Installation  (Mac | Docker). Aerospike Spring Integration.
pagetitle: Spring Boot Aerospike
summary: Aerospike is a key value datastore / Distributed Hashtable/NoSQL Database, providing features of in-memory NoSQL database. It merges the concept of in memory and No SQL database.
date: '2017-11-23'
update_date: '2019-09-08'
tags:
  - Spring
  - Aerospike
label:
  - Spring
slug: spring-aerospike
published: true
image: ../../common/atech-guide.png
---

## What is Aerospike

Aerospike is a key value datastore Or Distributed Hashtable Or NoSQL Database written in C language. It also provides features of in-memory database so its also called in-memory NoSQL database. It basically merges the concept of in memory and No SQL database. 


### Support for Hybrid Architecture.
Aerospikes' hybrid architecture helps us harness the power of, 
- Flash Memory
- Hard drives

### Layers
Aerospike consists of 3 layers
- **Data/Client layer**{: .heading1}  
  - Exposes API for storing and retrieving data.
  - Manages which node to be redirected to based on request coming from users.
- **Cluster layer**{: .heading1}    
  - Nodes are managed at this layer.
  - Maintains the consistency of data. 
- **Storage Layer**{: .heading1}    
  - Flash drives and D RAMS are handled in efficient way.
  - Basically Hybrid stuff lies here.

## Aerospike Installation

### MAC Installation
For vagrant installation follow [this](https://www.aerospike.com/docs/operations/install/vagrant/mac) URL

### Docker Installation
I've used docker to [install Aerospike](https://www.aerospike.com/docs/deploy_guides/docker) on my machine

#### Commands

```
docker run -d --name aerospike -p 3000:3000 aerospike/aerospike-server
```

<br/>

To bring up Management Console, run  

```java
docker run -d -p 8081:8081 mrbar42/aerospike-amc
```

<br/>

URL: http://localhost:8081/  
Then enter the IP of Aerospike container. To get IP use following command

```
docker inspect -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" container_id
```

## Aerospike Spring Integration

**Configuration**{: .heading1}

```java
@Configuration
@EnableAerospikeRepositories(basePackages = "in.kamranali.aerospike.aerospike.repositories")
@EnableTransactionManagement
public class AerospikeConfig {

    @Bean
    AerospikeTemplate aerospikeTemplate(){
        return new AerospikeTemplate(aerospikeClient(), "test"); // test is namespace
    }

    @Bean
    AerospikeClient aerospikeClient() {

        ClientPolicy clientPolicy = new ClientPolicy();
        clientPolicy.failIfNotConnected = true;
        return new AerospikeClient(clientPolicy, "localhost", 3000);
    }
}
```

<br/>

**UserResource**{: .heading1}

```java
@RestController
@RequestMapping("/rest/users")
public class UserResource {

    @Autowired
    private UserService userService;

    @GetMapping
    public List<User> getAllUsers(){

        return userService.getAllUsers();
    }

    @PostMapping
    public List<User> create(@RequestBody final User user){

        userService.create(user);
        return userService.getAllUsers();
    }
}
```

<br/>

**UserService**{: .heading1}

```java
@Service
public class UserService {

    @Autowired
    private UserRepository userRepository;

    public List<User> getAllUsers(){

        List<User> users = new ArrayList<>();
        userRepository.findAll()
                .forEach(users::add);
        return users;
    }

    public void create(User user) {

        userRepository.save(user);
    }
}
```

<br/>

**Repository**{: .heading1}

```java
public interface UserRepository extends AerospikeRepository<User, Integer> {
}
```

<br/>

**User Model**{: .heading1}

```java
public class User {

    @Id
    private Integer id;
    private String name;
    private Long salary;

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Long getSalary() {
        return salary;
    }

    public void setSalary(Long salary) {
        this.salary = salary;
    }
}
```

### Endpoints

GET http://localhost:8080/rest/users

POST http://localhost:8080/rest/users { "id": 1, "name": "Kamran", "salary": 1234 }

To access the full working code sample, click [here](https://github.com/kamranalinitb/springboot-blog/tree/master/aerospike){:target="_blank" rel="nofollow"}

## References
- [What is Aerospike? | Hybrid Architecture used by Flipkart | Tech Primers](https://www.youtube.com/watch?v=cf0-oXdChLY&t=380s){:target="_blank" rel="nofollow"}
