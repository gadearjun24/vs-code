// import fs from 'fs';
import {useState} from 'react'
import Video from "./components/Video";
import "./App.css";

function App() {

  var channel = [
    {
      title: "Mongodb Tutorial",
      views: "2k",
      verified: "true",
      channel_name: "Tech Savvy",
      image: "https://via.placeholder.com/600x400.png?text=Mongodb+Tutorial",
    },
    {
      title: "JavaScript Basics",
      views: "3k",
      verified: "true",
      channel_name: "Web Dev Simplified",
      image: "https://via.placeholder.com/600x400.png?text=JavaScript+Basics",
    },
    {
      title: "Python for Beginners",
      views: "5k",
      verified: "false",
      channel_name: "Programming with Mosh",
      image:
        "https://via.placeholder.com/600x400.png?text=Python+for+Beginners",
    },
    {
      title: "Advanced CSS",
      views: "1.2k",
      verified: "true",
      channel_name: "Traversy Media",
      image: "https://via.placeholder.com/600x400.png?text=Advanced+CSS",
    },
    {
      title: "React Tutorial",
      views: "8k",
      verified: "true",
      channel_name: "Academind",
      image: "https://via.placeholder.com/600x400.png?text=React+Tutorial",
    },
    {
      title: "Node.js Guide",
      views: "4k",
      verified: "false",
      channel_name: "The Net Ninja",
      image: "https://via.placeholder.com/600x400.png?text=Node.js+Guide",
    },
    {
      title: "SQL Queries",
      views: "6k",
      verified: "true",
      channel_name: "freeCodeCamp.org",
      image: "https://via.placeholder.com/600x400.png?text=SQL+Queries",
    },
    {
      title: "Git and GitHub",
      views: "7k",
      verified: "true",
      channel_name: "Traversy Media",
      image: "https://via.placeholder.com/600x400.png?text=Git+and+GitHub",
    },
    {
      title: "Java Programming",
      views: "2.5k",
      verified: "false",
      channel_name: "ProgrammingKnowledge",
      image: "https://via.placeholder.com/600x400.png?text=Java+Programming",
    },
    {
      title: "Web Development",
      views: "10k",
      verified: "true",
      channel_name: "The Net Ninja",
      image: "https://via.placeholder.com/600x400.png?text=Web+Development",
    },
    {
      title: "Machine Learning Intro",
      views: "9k",
      verified: "true",
      channel_name: "Sentdex",
      image:
        "https://via.placeholder.com/600x400.png?text=Machine+Learning+Intro",
    },
    {
      title: "API Development",
      views: "3.5k",
      verified: "false",
      channel_name: "Academind",
      image: "https://via.placeholder.com/600x400.png?text=API+Development",
    },
    {
      title: "Intro to Docker",
      views: "1.8k",
      verified: "true",
      channel_name: "TechWorld with Nana",
      image: "https://via.placeholder.com/600x400.png?text=Intro+to+Docker",
    },
    {
      title: "Cloud Computing",
      views: "5.5k",
      verified: "true",
      channel_name: "Simplilearn",
      image: "https://via.placeholder.com/600x400.png?text=Cloud+Computing",
    },
    {
      title: "Data Structures",
      views: "4.5k",
      verified: "true",
      channel_name: "MyCodeSchool",
      image: "https://via.placeholder.com/600x400.png?text=Data+Structures",
    },
    {
      title: "JavaScript ES6",
      views: "6.2k",
      verified: "false",
      channel_name: "The Net Ninja",
      image: "https://via.placeholder.com/600x400.png?text=JavaScript+ES6",
    },
    {
      title: "Kubernetes Basics",
      views: "2k",
      verified: "true",
      channel_name: "TechWorld with Nana",
      image: "https://via.placeholder.com/600x400.png?text=Kubernetes+Basics",
    },
    {
      title: "Intro to Algorithms",
      views: "7.3k",
      verified: "true",
      channel_name: "MIT OpenCourseWare",
      image: "https://via.placeholder.com/600x400.png?text=Intro+to+Algorithms",
    },
    {
      title: "Front-End Frameworks",
      views: "9.1k",
      verified: "false",
      channel_name: "Traversy Media",
      image:
        "https://via.placeholder.com/600x400.png?text=Front-End+Frameworks",
    },
    {
      title: "Cybersecurity 101",
      views: "4.8k",
      verified: "true",
      channel_name: "HackerSploit",
      image: "https://via.placeholder.com/600x400.png?text=Cybersecurity+101",
    },
    {
      title: "GraphQL Tutorial",
      views: "2.3k",
      verified: "true",
      channel_name: "Academind",
      image: "https://via.placeholder.com/600x400.png?text=GraphQL+Tutorial",
    },
    {
      title: "Swift Programming",
      views: "3.9k",
      verified: "false",
      channel_name: "CodeWithChris",
      image: "https://via.placeholder.com/600x400.png?text=Swift+Programming",
    },
    {
      title: "RESTful Services",
      views: "8.4k",
      verified: "true",
      channel_name: "ProgrammingKnowledge",
      image: "https://via.placeholder.com/600x400.png?text=RESTful+Services",
    },
    {
      title: "HTML & CSS",
      views: "5.8k",
      verified: "true",
      channel_name: "The Net Ninja",
      image: "https://via.placeholder.com/600x400.png?text=HTML+%26+CSS",
    },
    {
      title: "Database Design",
      views: "1.7k",
      verified: "true",
      channel_name: "freeCodeCamp.org",
      image: "https://via.placeholder.com/600x400.png?text=Database+Design",
    },
    {
      title: "Web Scraping",
      views: "2.9k",
      verified: "false",
      channel_name: "Tech With Tim",
      image: "https://via.placeholder.com/600x400.png?text=Web+Scraping",
    },
    {
      title: "Algorithm Optimization",
      views: "6.1k",
      verified: "true",
      channel_name: "GeeksforGeeks",
      image:
        "https://via.placeholder.com/600x400.png?text=Algorithm+Optimization",
    },
    {
      title: "Vue.js Essentials",
      views: "3.2k",
      verified: "true",
      channel_name: "Academind",
      image: "https://via.placeholder.com/600x400.png?text=Vue.js+Essentials",
    },
    {
      title: "Flutter Development",
      views: "7.6k",
      verified: "true",
      channel_name: "The Net Ninja",
      image: "https://via.placeholder.com/600x400.png?text=Flutter+Development",
    },
    {
      title: "AI and ML Trends",
      views: "5.2k",
      verified: "false",
      channel_name: "Two Minute Papers",
      image: "https://via.placeholder.com/600x400.png?text=AI+and+ML+Trends",
    },
    {
      title: "Security Best Practices",
      views: "4.1k",
      verified: "true",
      channel_name: "The Cyber Mentor",
      image:
        "https://via.placeholder.com/600x400.png?text=Security+Best+Practices",
    },
    {
      title: "DevOps Fundamentals",
      views: "2.8k",
      verified: "true",
      channel_name: "Simplilearn",
      image: "https://via.placeholder.com/600x400.png?text=DevOps+Fundamentals",
    },
    {
      title: "Typescript Guide",
      views: "3.7k",
      verified: "false",
      channel_name: "Academind",
      image: "https://via.placeholder.com/600x400.png?text=Typescript+Guide",
    },
  ];

  const [videos , setVideos] = useState([...channel])



function handleAddClick()
{
 var newData = {
    title: "Mongodb Tutorial",
    views: "2k",
    verified: "true",
    channel_name: "Tech Savvy",
    image: "https://via.placeholder.com/600x400.png?text=Mongodb+Tutorial",
  }

  setVideos([...channel , newData]);


}




  return (

    <>

<button onClick = {handleAddClick}>Add</button>
<br/>
      {videos.map((ele, index) => {
        return (
          <Video
            title={ele.title}
            views={ele.views}
            verified={ele.verified}
            key={index}
            image={ele.image}
            channel_name={ele.channel_name}
          ></Video>
        );
      })}
    </>
  );
}
export default App;
